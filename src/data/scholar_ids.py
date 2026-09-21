"""Import de Scholar ID saisis à la main (nom + ID) vers ``scholar_overrides.csv``.

Chaque nom est rapproché de la liste du PDF avec ``name_match_score`` ; un import n'est accepté que si le
rapprochement est fiable ET sans concurrent proche. Tout le reste est signalé, jamais deviné.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.data.scholar_matching import name_match_score

SCHOLAR_ID_RE = re.compile(r"^[A-Za-z0-9_-]{12}$")
OVERRIDE_FIELDS = ["chercheur_id", "scholar_id", "nom_complet", "laboratoire", "equipe"]


@dataclass
class ImportResult:
    """Sort d'une entrée du fichier fourni (``status`` explique pourquoi elle est, ou non, importée)."""

    input_name: str
    scholar_id: str
    status: str          # accepted | invalid_id | not_in_roster | ambiguous | other_institution | conflict
    chercheur_id: Optional[str] = None
    matched_name: Optional[str] = None
    etablissement: Optional[str] = None
    score: Optional[float] = None
    note: str = ""

    @property
    def to_verify(self) -> bool:
        """Accepté mais avec un nom qui ne correspond pas exactement (à relire)."""
        return self.status == "accepted" and (self.score or 0) < 0.95


def validate_scholar_id(value: str) -> Optional[str]:
    """Retourne ``None`` si l'ID a le format Scholar (12 caractères ``[A-Za-z0-9_-]``), sinon la raison."""
    if not value:
        return "ID vide"
    if SCHOLAR_ID_RE.match(value):
        return None
    return f"format invalide ({len(value)} caractères, 12 attendus) — recopiez l'ID depuis l'URL du profil"


def match_entries(entries: list[dict[str, Any]], roster: list[dict[str, Any]], institution: str = "FSBM",
                  min_score: float = 0.80, min_margin: float = 0.10) -> list[ImportResult]:
    """Rapproche chaque entrée ``{nom_complet, chercheur_id (= Scholar ID)}`` de la liste ``roster``.

    ``roster`` contient TOUS les membres du PDF (pas seulement l'établissement cible) afin qu'une personne
    d'un autre établissement ne soit pas rapprochée à tort d'un homonyme approximatif de la FSBM.
    """
    results: list[ImportResult] = []
    seen: dict[str, str] = {}
    for entry in entries:
        name, sid = (entry.get("nom_complet") or "").strip(), (entry.get("chercheur_id") or "").strip()
        scored = sorted(((name_match_score(name, r["nom_complet"]), r) for r in roster), key=lambda t: t[0], reverse=True)
        best_score, best = scored[0] if scored else (0.0, None)
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        res = ImportResult(name, sid, "accepted", best["chercheur_id"] if best else None,
                           best["nom_complet"] if best else None, best.get("etablissement") if best else None, round(best_score, 3))
        if best is None or best_score < min_score:
            res.status, res.chercheur_id = "not_in_roster", None
            res.note = f"aucun membre du PDF ne correspond (plus proche : {res.matched_name}, score {best_score:.2f})"
        elif best_score - second_score < min_margin:
            res.status = "ambiguous"
            res.note = f"plusieurs membres possibles (scores {best_score:.2f} et {second_score:.2f})"
        elif (best.get("etablissement") or "").strip().upper() != institution.upper():
            res.status = "other_institution"
            res.note = f"membre {best.get('etablissement')} : hors périmètre {institution}"
        elif reason := validate_scholar_id(sid):
            res.status, res.note = "invalid_id", reason
        elif res.chercheur_id in seen and seen[res.chercheur_id] != sid:
            res.status, res.note = "conflict", f"deux ID différents pour ce chercheur ({seen[res.chercheur_id]} / {sid})"
        else:
            seen[res.chercheur_id] = sid
        results.append(res)
    return results


def read_overrides(path: Path) -> dict[str, str]:
    """Lit ``chercheur_id → scholar_id`` (lignes sans ID ignorées)."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["chercheur_id"].strip(): (r.get("scholar_id") or "").strip()
                for r in csv.DictReader(fh) if r.get("chercheur_id")}


def write_overrides(records: list[dict[str, Any]], path: Path, new_ids: Optional[dict[str, str]] = None) -> int:
    """(Ré)écrit ``scholar_overrides.csv`` : un chercheur par ligne, ID existants conservés, ``new_ids`` ajoutés.

    Retourne le nombre de lignes avec un ID renseigné.
    """
    ids = {**read_overrides(path), **(new_ids or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OVERRIDE_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({"chercheur_id": rec["chercheur_id"], "scholar_id": ids.get(rec["chercheur_id"], ""),
                             "nom_complet": rec["nom_complet"], "laboratoire": rec.get("laboratoire") or "",
                             "equipe": rec.get("equipe") or ""})
    return sum(1 for rec in records if ids.get(rec["chercheur_id"]))
