import os
import random
from pathlib import Path
import numpy as np


class AppConfig:
    SEED = 42
    TEST_SIZE = 0.2

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        RAW_DATA_DIR = APP_HOME / "data"
        RAW_DATA_FILE = RAW_DATA_DIR / "spam.csv"
        AUGMENTED_DATA_FILE=RAW_DATA_DIR / "spam_augmented.csv"


def seed_everything(seed: int = AppConfig.SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
