"""Central configuration and repository-relative paths for the pipeline."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "data"
DATA_PATH = DATA_DIR / "btc_processed.csv"
MODEL_DIR = Path(__file__).resolve().parent / "model_weight"
MODEL_PATH = MODEL_DIR / "lstm_model.pth"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_PATH = PLOT_DIR / "cumulative_comparison.png"

SYMBOL = "BTC-USD"
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"
INTERVAL = "1d"

FEATURE_COLUMNS = (
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MA_7",
    "MA_30",
    "Volatility",
)
TARGET_COLUMN = "Target"
FUTURE_RETURN_COLUMN = "Future_Return"
TARGET_THRESHOLD = 0.005

TRAIN_SPLIT = 0.8
SEQUENCE_LENGTH = 30
UPPER_SIGNAL_THRESHOLD = 0.52
LOWER_SIGNAL_THRESHOLD = 0.48

INPUT_SIZE = len(FEATURE_COLUMNS)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
