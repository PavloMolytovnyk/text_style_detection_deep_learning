from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLES = ["academic", "business", "formal", "informal", "literary"]
TONES = ["aggressive", "friendly", "neutral", "sarcastic", "urgent"]
SEED = 42
MAX_CHARS = 8000
DEFAULT_DATA = ROOT / "data" / "prepared"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
