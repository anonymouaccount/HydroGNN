import pandas as pd
import numpy as np

from config import (
    NODE_ID_COL,
    SOURCE_COL,
    TARGET_COL,
    REMOVE_DUPLICATES,
    FILL_NUMERIC_WITH,
    FILL_CATEGORICAL_WITH,
    MIN_SLOPE,
    MIN_DEPTH,
    MAX_DIAMETER
)


# =========================
# LOAD DATA
# =========================
def load_data(nodes_path, arcs_path, basins_path):
    df_nodes = pd.read_csv(nodes_path, low_memory=False)
    df_arcs = pd.read_csv(arcs_path, low_memory=False)
    df_basins = pd.read_csv(basins_path, low_memory=False)

    return df_nodes, df_arcs, df_basins


# =========================
# MAIN CLEAN FUNCTION
# =========================
def clean_data(df_nodes, df_arcs, df_basins):
    print("   → Cleaning nodes...")
    df_nodes = clean_nodes(df_nodes)

    print("   → Cleaning arcs...")
    df_arcs = clean_arcs(df_arcs)

    print("   → Cleaning basins...")
    df_basins = clean_basins(df_basins)

    return df_nodes, df_arcs, df_basins


# =========================
# CLEAN NODES
# =========================
def clean_nodes(df):
    df = df.copy()

    # --- Remove duplicates ---
    if REMOVE_DUPLICATES and NODE_ID_COL in df.columns:
        df = df.sort_values(by=df.columns.tolist())  # stable order
        df = df.drop_duplicates(subset=[NODE_ID_COL], keep="first")

    # --- Handle missing values ---
    df = fill_missing_values(df)

    return df


# =========================
# CLEAN ARCS (EDGES)
# =========================
def clean_arcs(df):
    df = df.copy()

    # Replace invalid IDs (0 → NaN)
    if SOURCE_COL in df.columns:
        df[SOURCE_COL] = df[SOURCE_COL].replace(0, np.nan)

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].replace(0, np.nan)

    # --- Remove duplicates (optional) ---
    if REMOVE_DUPLICATES:
        df = df.drop_duplicates()

    # --- Fix anomalies ---
    if "slope" in df.columns:
        df.loc[df["slope"] < MIN_SLOPE, "slope"] = np.nan

    if "depth" in df.columns:
        df.loc[df["depth"] < MIN_DEPTH, "depth"] = np.nan

    if "diameter" in df.columns:
        df.loc[df["diameter"] > MAX_DIAMETER, "diameter"] = np.nan

    # --- Handle missing values ---
    df = fill_missing_values(df)

    return df


# =========================
# CLEAN BASINS
# =========================
def clean_basins(df):
    df = df.copy()

    if REMOVE_DUPLICATES:
        df = df.drop_duplicates()

    df = fill_missing_values(df)

    return df


# =========================
# MISSING VALUE HANDLING
# =========================
def fill_missing_values(df):
    df = df.copy()

    for col in df.columns:

        # -------------------------
        # NUMERIC COLUMNS
        # -------------------------
        if pd.api.types.is_numeric_dtype(df[col]):

            if df[col].isnull().sum() > 0:

                if FILL_NUMERIC_WITH == "mean":
                    df[col] = df[col].fillna(df[col].mean())

                elif FILL_NUMERIC_WITH == "median":
                    df[col] = df[col].fillna(df[col].median())

                elif FILL_NUMERIC_WITH == "zero":
                    df[col] = df[col].fillna(0)

        # -------------------------
        # CATEGORICAL COLUMNS
        # -------------------------
        else:

            if df[col].isnull().sum() > 0:

                if FILL_CATEGORICAL_WITH == "mode":

                    mode_values = df[col].mode()

                    if len(mode_values) > 0:
                        df[col] = df[col].fillna(mode_values[0])
                    else:
                        df[col] = df[col].fillna("unknown")

                else:
                    df[col] = df[col].fillna("unknown")

    return df