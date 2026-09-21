"""Rend le paquet ``src`` importable quand un script est lancé directement (python scripts/xx.py)."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# scholarly émet des SyntaxWarning à l'import (regex non « raw ») : bruit sans conséquence
warnings.filterwarnings("ignore", category=SyntaxWarning)
