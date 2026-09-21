"""Étape 0 — Extraction des chercheurs depuis « Membres FSBM.pdf » → data/raw/chercheurs_fsbm.csv.

Exemples :
    python scripts/01_extract_members.py
    python scripts/01_extract_members.py --pdf "data/input/Membres FSBM.pdf" --inspect
"""
import argparse
import sys
from pathlib import Path

from _bootstrap import ROOT

from src.data.extract_fsbm_members import inspect_pdf, run_extraction, summarize_members
from src.utils.config import load_settings, resolve_path
from src.utils.io import write_json
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", type=Path, help="Chemin du PDF (défaut : paths.members_pdf de settings.yaml)")
    p.add_argument("--output", type=Path, help="CSV de sortie (défaut : data/raw/chercheurs_fsbm.csv)")
    p.add_argument("--inspect", action="store_true", help="Affiche la mise en page brute du PDF puis s'arrête (diagnostic)")
    p.add_argument("--forward-fill", nargs="*", default=None, metavar="COL",
                   help="Colonnes à propager vers le bas (cellules fusionnées) : etablissement laboratoire equipe type_membre")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def locate_pdf(explicit: Path | None, default: Path) -> Path:
    """PDF explicite, sinon chemin par défaut, sinon recherche d'un fichier « Membres*FSBM*.pdf »."""
    for candidate in (explicit, default):
        if candidate and candidate.exists():
            return candidate
    for folder in (ROOT, ROOT / "data", ROOT / "data" / "input", ROOT.parent):
        for match in sorted(folder.glob("Membres*FSBM*.pdf")):
            return match
    raise FileNotFoundError(f"PDF introuvable. Placez « Membres FSBM.pdf » dans {default.parent} ou utilisez --pdf.")


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log = setup_logging("01_extract_members", resolve_path(settings, "logs_dir"))
    try:
        pdf = locate_pdf(args.pdf, resolve_path(settings, "members_pdf"))
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    log.info("PDF source : %s", pdf)
    if args.inspect:
        print(inspect_pdf(pdf))
        return 0

    forward_fill = args.forward_fill if args.forward_fill is not None else settings["data"].get("forward_fill_columns", [])
    df, report = run_extraction(pdf, forward_fill)
    out = args.output or resolve_path(settings, "raw_dir") / "chercheurs_fsbm.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    stats = summarize_members(df, settings["data"]["institution"])
    reports_dir = resolve_path(settings, "reports_dir")
    write_json(reports_dir / "members_extraction_report.json", {"summary": stats, "extraction": report.__dict__})
    if report.unparsed_lines:
        (reports_dir / "members_unparsed_lines.txt").write_text("\n".join(report.unparsed_lines), encoding="utf-8")

    print("\n=== Extraction du PDF Membres FSBM ===")
    print(f"Méthode d'extraction        : {report.method} ({report.pages} pages)")
    print(f"Lignes extraites            : {stats['lignes_extraites']}")
    print(f"Personnes uniques (total)   : {stats['personnes_uniques_total']}")
    print(f"Membres {settings['data']['institution']} (uniques)      : {stats['membres_fsbm_uniques']}")
    print(f"Établissements              : {stats['etablissements']}")
    print(f"Laboratoires ({settings['data']['institution']})       : {stats['laboratoires_fsbm']}")
    print(f"Équipes ({settings['data']['institution']})            : {stats['equipes_fsbm']}")
    print(f"Lignes doublons exactes     : {stats['lignes_doublons_exacts']}")
    print(f"Personnes multi-affectation : {stats['personnes_avec_plusieurs_affectations']}")
    print(f"Valeurs manquantes          : {stats['valeurs_manquantes']}")
    print("\nChercheurs par laboratoire :")
    for lab, n in stats["chercheurs_par_laboratoire"].items():
        print(f"  {n:4d}  {lab}")
    print("\nChercheurs par équipe :")
    for team, n in stats["chercheurs_par_equipe"].items():
        print(f"  {n:4d}  {team}")
    if report.warnings or report.unparsed_lines or report.skipped:
        print(f"\n⚠ {len(report.warnings)} avertissement(s), {len(report.unparsed_lines)} ligne(s) non analysée(s), "
              f"{len(report.skipped)} ligne(s) ignorée(s) — voir {reports_dir}")
    log.info("Fichier écrit : %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
