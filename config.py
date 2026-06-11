import os

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# =========================
# DATA SETTINGS
# =========================
NODE_ID_COL = "id_node"
SOURCE_COL = "f_id_node"
TARGET_COL = "t_id_node"

# Columns you expect (you can adapt later)
NODE_FEATURES = ["coordx", "coordy", "z", "depth"]
EDGE_FEATURES = ["length", "diameter", "slope"]

# =========================
# CLEANING SETTINGS
# =========================
REMOVE_DUPLICATES = True

# Thresholds for anomaly filtering
MIN_SLOPE = 0
MIN_DEPTH = 0
MAX_DIAMETER = 3000  # mm (adapt if needed)

# Missing value strategy
FILL_NUMERIC_WITH = "mean"   # options: mean, median, zero
FILL_CATEGORICAL_WITH = "mode"

# =========================
# GRAPH SETTINGS
# =========================
K_NEIGHBORS = 3   # for spatial reconstruction if topology missing

# =========================
# MODEL SETTINGS
# =========================
MODEL_TYPE = "GCN"   # or "GraphSAGE", "GAT"

# =========================
# MODEL DIMENSIONS
# =========================
INPUT_DIM = len(NODE_FEATURES)

HIDDEN_DIM = 64

OUTPUT_DIM = 32

NUM_HEADS = 4

DROPOUT = 0.2
# =========================
# TRAINING SETTINGS
# =========================
EPOCHS = 300

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5


# =========================
# LOSS WEIGHTS
# =========================
LAMBDA_REC = 1.0

LAMBDA_LINK = 0.5

LAMBDA_PHYS = 0.1


# =========================
# GRAPH
# =========================
K_NEIGHBORS = 3

TEST_EDGE_RATIO = 0.2


# =========================
# GAT
# =========================
HIDDEN_DIM = 64

OUTPUT_DIM = 32

DROPOUT = 0.2

# =========================
# DEVICE
# =========================
DEVICE = "cpu"  # change to "cpu" if needed