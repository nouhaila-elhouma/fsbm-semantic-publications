"""Résolution prudente du profil Google Scholar d'un chercheur FSBM.

Principe : le nom seul ne suffit JAMAIS. Le score combine nom, affiliation (FSBM /
Université Hassan II / Casablanca), domaine d'email institutionnel, cohérence avec le
laboratoire et — après récupération du profil — co-auteurs appartenant à la liste FSBM.
En cas d'ambiguïté, on ne choisit pas : les candidats sont conservés pour vérification manuelle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

from src.preprocessing.text_cleaner import normalize_for_matching

PARTICLES = {"el", "al"}
FSBM_RE = re.compile(r"\bben\s?m\s?sik\b|\bfsbm\b|\bfaculty of sciences? ben\b|\bfacult\w* des sciences ben\b")
UNIVERSITY_RE = re.compile(r"\bhassan\s?(ii|2)\b|\buniv\s?h2c\b|\bh2c\b")
CASABLANCA_RE = re.compile(r"\bcasablanca\b")
INSTITUTIONAL_DOMAIN = "univh2c.ma"
STOPWORDS = {"de", "des", "du", "la", "le", "et", "and", "of", "the", "des", "en", "d", "l", "laboratoire",
             "laboratory", "equipe", "team", "group", "groupe", "sciences", "science", "recherche", "research"}

WEIGHTS = {"name": 0.40, "fsbm": 0.25, "university": 0.15, "email": 0.10, "casablanca": 0.05, "topic": 0.05}


# --------------------------------------------------------------------------- noms
def _tokens(name: Optional[str]) -> list[str]:
    return normalize_for_matching(name).split()


def name_match_score(a: Optional[str], b: Optional[str]) -> float:
    """Similarité de deux noms de personnes, insensible à l'ordre, aux accents et aux
    variantes d'espacement (« Ben Lahmar » ≈ « Benlahmar »), avec gestion des initiales.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    if set(ta) == set(tb):
        return 1.0
    # mêmes lettres (anagramme) => même nom écrit avec un découpage différent : « Elhabib Benlahmar »
    # ≈ « El Habib Ben Lahmar », « Aitdaoud » ≈ « Ait Daoud » ; avec ou sans particules « el »/« al »
    for drop in (set(), PARTICLES):
        core_a = "".join(sorted("".join(t for t in ta if t not in drop)))
        core_b = "".join(sorted("".join(t for t in tb if t not in drop)))
        if core_a and core_a == core_b:
            return 0.95
    # particule « el »/« al » collée au prénom (« Elhabib » ≈ « Habib ») : on la retire des deux côtés
    def _strip_particles(tokens: list[str]) -> str:
        kept = [t[2:] if t[:2] in PARTICLES and len(t) > 4 else t for t in tokens if t not in PARTICLES]
        return "".join(sorted("".join(kept)))

    if _strip_particles(ta) and _strip_particles(ta) == _strip_particles(tb):
        return 0.93
    # un nom est entièrement contenu dans l'autre (ex. nom d'épouse ajouté) : plausible mais à vérifier
    short, long_ = (set(ta), set(tb)) if len(ta) <= len(tb) else (set(tb), set(ta))
    if len(short) >= 2 and short < long_:
        return 0.85
    # recouvrement de tokens, une initiale valant 0.75 token
    remaining = list(tb)
    matched = 0.0
    for tok in ta:
        if tok in remaining:
            remaining.remove(tok)
            matched += 1
            continue
        for other in remaining:
            if (len(tok) == 1 and other.startswith(tok)) or (len(other) == 1 and tok.startswith(other)):
                remaining.remove(other)
                matched += 0.75
                break
    overlap = matched / max(len(ta), len(tb))
    fuzzy = SequenceMatcher(None, "".join(sorted(ta)), "".join(sorted(tb))).ratio()
    return max(overlap, 0.9 * fuzzy if fuzzy >= 0.85 else 0.0)


# --------------------------------------------------------------------------- scoring
def _topic_overlap(context: Optional[str], candidate_text: str) -> bool:
    ctx = {t for t in _tokens(context) if len(t) > 3 and t not in STOPWORDS}
    cand = set(_tokens(candidate_text))
    return bool(ctx & cand)


@dataclass
class ScoredCandidate:
    """Candidat Scholar avec son score et le détail des indices retenus."""

    candidate: dict[str, Any]
    score: float
    name_score: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**self.candidate, "score": round(self.score, 3), "name_score": round(self.name_score, 3),
                "evidence": self.evidence}


def score_candidate(researcher: dict[str, Any], candidate: dict[str, Any]) -> ScoredCandidate:
    """Score de confiance ∈ [0, 1] (le nom seul plafonne à 0,40)."""
    name_score = name_match_score(researcher.get("nom_complet"), candidate.get("name"))
    affiliation = normalize_for_matching(candidate.get("affiliation"))
    domain = (candidate.get("email_domain") or "").lower()
    interests = " ".join(candidate.get("interests") or [])
    lab_context = " ".join(filter(None, [researcher.get("laboratoire"), researcher.get("equipe")]))

    evidence = {
        "fsbm": bool(FSBM_RE.search(affiliation)),
        "university": bool(UNIVERSITY_RE.search(affiliation)) or domain.endswith(INSTITUTIONAL_DOMAIN),
        "email": domain.endswith(INSTITUTIONAL_DOMAIN),
        "casablanca": bool(CASABLANCA_RE.search(affiliation)),
        "topic": _topic_overlap(lab_context, f"{affiliation} {interests}"),
    }
    score = WEIGHTS["name"] * name_score + sum(WEIGHTS[k] for k, present in evidence.items() if present)
    return ScoredCandidate(candidate, min(1.0, score), name_score, evidence)


def coauthor_bonus(coauthors: Iterable[str], roster_names: Iterable[str], self_name: str) -> tuple[float, list[str]]:
    """Bonus (≤ 0,10) si des co-auteurs du profil figurent dans la liste FSBM (hors le chercheur lui-même)."""
    roster = [n for n in roster_names if name_match_score(n, self_name) < 0.9]
    hits = [c for c in coauthors if any(name_match_score(c, n) >= 0.95 for n in roster)]
    return min(0.10, 0.05 * len(hits)), hits


# --------------------------------------------------------------------------- décision
@dataclass
class MatchDecision:
    """Résultat de la résolution : statut, candidat retenu (si sûr) et tous les candidats notés."""

    status: str                      # matched | ambiguous | not_found
    confidence: Optional[float]
    scholar_id: Optional[str]
    detail: str
    candidates: list[dict[str, Any]]
    method: str = "auto_scoring"


def decide_match(researcher: dict[str, Any], candidates: list[dict[str, Any]], *, accept: float = 0.70,
                 ambiguous: float = 0.50, min_margin: float = 0.15, min_name_score: float = 0.66) -> MatchDecision:
    """Choisit un profil uniquement s'il est à la fois fiable et sans concurrent proche."""
    scored = sorted((score_candidate(researcher, c) for c in candidates), key=lambda s: s.score, reverse=True)
    plausible = [s for s in scored if s.name_score >= min_name_score]
    listing = [s.to_dict() for s in scored[:5]]
    if not plausible:
        return MatchDecision("not_found", None, None,
                             "aucun candidat dont le nom correspond" if scored else "aucun résultat de recherche", listing)
    best = plausible[0]
    runner_up = plausible[1] if len(plausible) > 1 else None
    margin_ok = runner_up is None or (best.score - runner_up.score) >= min_margin
    if best.score >= accept and margin_ok:
        return MatchDecision("matched", round(best.score, 3), best.candidate["scholar_id"],
                             f"score {best.score:.2f}, preuves : {[k for k, v in best.evidence.items() if v]}", listing)
    if best.score >= ambiguous:
        reason = "plusieurs profils plausibles" if not margin_ok else "confiance insuffisante"
        return MatchDecision("ambiguous", round(best.score, 3), None,
                             f"{reason} (meilleur score {best.score:.2f}) — vérification manuelle requise", listing)
    return MatchDecision("not_found", round(best.score, 3), None,
                         f"meilleur candidat trop peu fiable ({best.score:.2f})", listing)
