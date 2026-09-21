"""Lecture et application de ``robots.txt`` (sémantique Google : jokers ``*``, ``$``, règle la plus longue prioritaire).

Le module standard ``urllib.robotparser`` ne gère pas les jokers ``*`` dans les chemins, or Google Scholar
les utilise (``Disallow: /citations?*cstart=``) : d'où cette implémentation minimale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


class RobotsDisallowedError(RuntimeError):
    """L'URL demandée est interdite par robots.txt : la requête n'est PAS envoyée."""


class SearchNotAllowedError(RobotsDisallowedError):
    """La recherche automatique d'auteurs par nom est désactivée (interdite par robots.txt de Google Scholar)."""


@dataclass
class _Rule:
    allow: bool
    pattern: str
    regex: re.Pattern[str]


@dataclass
class RobotsRules:
    """Règles d'un groupe ``User-agent`` ; ``is_allowed`` reçoit « chemin?requête »."""

    rules: list[_Rule] = field(default_factory=list)

    @staticmethod
    def _compile(pattern: str) -> re.Pattern[str]:
        anchored = pattern.endswith("$")
        body = re.escape(pattern.rstrip("$")).replace(r"\*", ".*")
        return re.compile("^" + body + ("$" if anchored else ""))

    @classmethod
    def from_text(cls, text: str, agent: str = "*") -> "RobotsRules":
        """Extrait les règles du groupe qui s'applique à ``agent`` (à défaut, au groupe ``*``)."""
        groups: list[tuple[set[str], list[tuple[bool, str]]]] = []
        agents: set[str] = set()
        rules: list[tuple[bool, str]] = []
        in_rules = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if in_rules:                       # nouveau groupe
                    groups.append((agents, rules))
                    agents, rules, in_rules = set(), [], False
                agents.add(value.lower())
            elif key in {"allow", "disallow"}:
                in_rules = True
                if value:                          # « Disallow: » vide = tout autorisé
                    rules.append((key == "allow", value))
        groups.append((agents, rules))
        chosen = next((r for a, r in groups if agent.lower() in a), None)
        if chosen is None:
            chosen = next((r for a, r in groups if "*" in a), [])
        return cls([_Rule(allow, pat, cls._compile(pat)) for allow, pat in chosen])

    def matching_rule(self, target: str) -> Optional[_Rule]:
        """Règle applicable : la plus longue ; à longueur égale, ``Allow`` l'emporte."""
        best: Optional[_Rule] = None
        for rule in self.rules:
            if rule.regex.match(target):
                if best is None or len(rule.pattern) > len(best.pattern) or (
                        len(rule.pattern) == len(best.pattern) and rule.allow and not best.allow):
                    best = rule
        return best

    def is_allowed(self, target: str) -> bool:
        rule = self.matching_rule(target)
        return True if rule is None else rule.allow
