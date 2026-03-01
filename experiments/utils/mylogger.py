import logging
from pathlib import Path

RESULTS_DIR = Path("results")
LOG_FILE = RESULTS_DIR / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

class MrSQMFilter(logging.Filter):
    _noise = {
        "Filter subsequences",
        "Random sampling",
        "Symbolic Parameters",
        "Sampling window size",
        "Found ",
        "Fit training data",
        "Search for subsequences",
        "Transform time series",
        "Select ",
        "Compute ",
    }
    def filter(self, record):
        return not any(record.getMessage().startswith(p) for p in self._noise)

# Apply to root logger so it catches the pyx calls
logging.getLogger().addFilter(MrSQMFilter())