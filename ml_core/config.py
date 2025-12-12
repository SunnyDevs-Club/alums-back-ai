from pathlib import Path

class Config:
    PROJ_DIR: Path = Path(__file__).resolve().parent

    # Model & Data Loader Initializing
    DATASET_DIR: Path = PROJ_DIR / 'data' / 'inference'
    WEIGHT_DIR: Path = PROJ_DIR / 'weights'

    FOLD_NUM = 1
    NUM_WORKERS = 4
    DEVICE = 'cpu'
    BATCH_SIZE = 32
    N_PIXEL = 64

    INPUT_DIM = 10
    MLP1 = [10, 32, 64]
    POOLING = 'mean_std'
    MLP2 = [132, 128]
    GEOMETRIC_FEATURES = 1

    N_HEAD = 4
    D_K = 32
    MLP3 = [512, 128, 128]
    DROPOUT = 0.3
    T = 1000
    LMS = 100

    NUM_CLASSES = 3
    MLP4 = [128, 64, 32, 3]

    # Preprocessing
    STAC_SERVICE_URL = "https://earth-search.aws.element84.com/v1"
    COLLECTION = ['sentinel-2-l2a']
    PLATFORM = "sentinel-2b"
    RES_10M_BANDS = ["blue", "green", "red", "nir"]
    RES_20M_BANDS = ["rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]

    DATE_RANGE = '2024-02-01/2024-09-30'

    # Define the mapping from crop types to labels
    CROP_LABELS = {
        "Paxta": 0,
        "G'alla": 1,
        "Bog'": 2,
    }

    # Reverse mapping from labels to crop types
    LABEL_TO_CROP = {v: k for k, v in CROP_LABELS.items()}

