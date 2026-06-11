import numpy as np
import pandas as pd
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import (
    NODE_ID_COL,
    SOURCE_COL,
    TARGET_COL,
    NODE_FEATURES,
    EDGE_FEATURES,
    K_NEIGHBORS,
    TEST_EDGE_RATIO
)


# =========================
# MAIN GRAPH PIPELINE
# =========================
def build_graph(
    df_nodes,
    df_arcs,
    df_basins=None
):

    print("   → Preparing node features...")

    node_features, node_id_map = prepare_node_features(
        df_nodes
    )

    print("   → Preparing edge index...")

    edge_index = build_edge_index(
        df_nodes,
        df_arcs,
        node_id_map
    )

    print("   → Splitting edges...")

    train_edges, test_edges = split_edges(
        edge_index,
        test_ratio=TEST_EDGE_RATIO
    )

    print("   → Preparing edge features...")

    edge_attr = prepare_edge_features(
        df_arcs
    )

    print("   → Creating mask...")

    mask = create_mask(
        node_features
    )

    # -------------------------
    # FINAL SAFETY
    # -------------------------
    node_features = np.nan_to_num(
        node_features,
        nan=0.0
    )

    edge_attr = np.nan_to_num(
        edge_attr,
        nan=0.0
    )

    print("   → Converting to tensors...")

    data = {

        # Node features
        "x": torch.tensor(
            node_features,
            dtype=torch.float
        ),

        # Training edges
        "train_edge_index": torch.tensor(
            train_edges,
            dtype=torch.long
        ),

        # Test edges
        "test_edge_index": torch.tensor(
            test_edges,
            dtype=torch.long
        ),

        # Full graph
        "edge_index": torch.tensor(
            edge_index,
            dtype=torch.long
        ),

        # Edge attributes
        "edge_attr": torch.tensor(
            edge_attr,
            dtype=torch.float
        ),

        # Missing value mask
        "mask": torch.tensor(
            mask,
            dtype=torch.float
        )
    }

    return data


# =========================
# NODE FEATURES
# =========================
def prepare_node_features(df_nodes):

    df = df_nodes.copy().reset_index(
        drop=True
    )

    # -------------------------
    # ENSURE FEATURES EXIST
    # -------------------------
    for col in NODE_FEATURES:

        if col not in df.columns:
            df[col] = 0

    # -------------------------
    # SAFE PROCESSING
    # -------------------------
    features = df[NODE_FEATURES].copy()

    for col in NODE_FEATURES:

        features[col] = pd.to_numeric(
            features[col],
            errors="coerce"
        )

    # Fill missing
    features = features.fillna(
        features.mean()
    )

    # Standardization
    scaler = StandardScaler()

    features = scaler.fit_transform(
        features
    )

    # Final protection
    features = np.nan_to_num(
        features
    )

    # -------------------------
    # NODE ID MAP
    # -------------------------
    node_ids = df[
        NODE_ID_COL
    ].values

    node_id_map = {

        node_id: idx

        for idx, node_id
        in enumerate(node_ids)
    }

    return features, node_id_map


# =========================
# EDGE INDEX
# =========================
def build_edge_index(
    df_nodes,
    df_arcs,
    node_id_map
):

    edges = []

    # -------------------------
    # EXISTING TOPOLOGY
    # -------------------------
    for _, row in df_arcs.iterrows():

        src = row.get(SOURCE_COL)

        tgt = row.get(TARGET_COL)

        if pd.notna(src) and pd.notna(tgt):

            if (
                src in node_id_map
                and tgt in node_id_map
            ):

                edges.append([
                    node_id_map[src],
                    node_id_map[tgt]
                ])

    # -------------------------
    # FALLBACK KNN
    # -------------------------
    if len(edges) < len(df_nodes):

        print(
            "   Incomplete topology detected "
            "→ using KNN reconstruction..."
        )

        edges = reconstruct_edges_knn(
            df_nodes,
            node_id_map
        )

    edge_index = np.array(edges).T

    return edge_index


# =========================
# KNN TOPOLOGY
# =========================
def reconstruct_edges_knn(
    df_nodes,
    node_id_map,
    k=K_NEIGHBORS
):

    print(
        "   → Reconstructing topology with KNN..."
    )

    # -------------------------
    # COORDINATE COLUMNS
    # -------------------------
    possible_x = [
        "x",
        "coordx",
        "coord_x",
        "longitude",
        "lon"
    ]

    possible_y = [
        "y",
        "coordy",
        "coord_y",
        "latitude",
        "lat"
    ]

    x_col = None
    y_col = None

    lower_cols = {
        c.lower(): c
        for c in df_nodes.columns
    }

    for col in possible_x:

        if col.lower() in lower_cols:

            x_col = lower_cols[
                col.lower()
            ]

            break

    for col in possible_y:

        if col.lower() in lower_cols:

            y_col = lower_cols[
                col.lower()
            ]

            break

    if x_col is None or y_col is None:

        raise ValueError(
            "Coordinate columns not found."
        )

    # -------------------------
    # SAFE NUMERIC CONVERSION
    # -------------------------
    df_temp = df_nodes.copy()

    df_temp[x_col] = (
        df_temp[x_col]
        .astype(str)
        .str.replace(",", ".")
    )

    df_temp[y_col] = (
        df_temp[y_col]
        .astype(str)
        .str.replace(",", ".")
    )

    df_temp[x_col] = pd.to_numeric(
        df_temp[x_col],
        errors="coerce"
    )

    df_temp[y_col] = pd.to_numeric(
        df_temp[y_col],
        errors="coerce"
    )

    # -------------------------
    # REMOVE INVALID ROWS
    # -------------------------
    valid_nodes = df_temp.dropna(
        subset=[x_col, y_col]
    ).copy()

    valid_nodes = valid_nodes[
        valid_nodes[NODE_ID_COL]
        .isin(node_id_map)
    ]

    print(
        f"   → Valid spatial nodes: "
        f"{len(valid_nodes)}"
    )

    coords = valid_nodes[
        [x_col, y_col]
    ].values.astype(float)

    coords = coords[
        np.isfinite(coords).all(axis=1)
    ]

    coords = np.nan_to_num(
        coords
    )

    if len(coords) == 0:

        raise ValueError(
            "No valid coordinates available."
        )

    # -------------------------
    # KNN
    # -------------------------
    nbrs = NearestNeighbors(
        n_neighbors=min(
            k + 1,
            len(coords)
        )
    )

    nbrs.fit(coords)

    distances, indices = nbrs.kneighbors(
        coords
    )

    row_to_node_idx = [

        node_id_map[nid]

        for nid in valid_nodes[
            NODE_ID_COL
        ].values
    ]

    edges = []

    for i, neighbors in enumerate(indices):

        src_idx = row_to_node_idx[i]

        for j in neighbors[1:]:

            tgt_idx = row_to_node_idx[j]

            edges.append([
                src_idx,
                tgt_idx
            ])

    print(
        f"   → Reconstructed edges: "
        f"{len(edges)}"
    )

    return edges


# =========================
# EDGE SPLIT
# =========================
def split_edges(
    edge_index,
    test_ratio=0.2
):

    num_edges = edge_index.shape[1]

    indices = np.arange(num_edges)

    np.random.shuffle(indices)

    split_idx = int(
        num_edges * (1 - test_ratio)
    )

    train_idx = indices[:split_idx]

    test_idx = indices[split_idx:]

    train_edges = edge_index[
        :,
        train_idx
    ]

    test_edges = edge_index[
        :,
        test_idx
    ]

    print(
        f"   → Train edges: "
        f"{train_edges.shape[1]}"
    )

    print(
        f"   → Test edges: "
        f"{test_edges.shape[1]}"
    )

    return train_edges, test_edges


# =========================
# EDGE FEATURES
# =========================
def prepare_edge_features(df_arcs):

    df = df_arcs.copy()

    for col in EDGE_FEATURES:

        if col not in df.columns:
            df[col] = 0

    features = df[
        EDGE_FEATURES
    ].copy()

    for col in EDGE_FEATURES:

        features[col] = pd.to_numeric(
            features[col],
            errors="coerce"
        )

    features = features.fillna(
        features.mean()
    )

    scaler = StandardScaler()

    features = scaler.fit_transform(
        features
    )

    features = np.nan_to_num(
        features
    )

    return features


# =========================
# MASK
# =========================
def create_mask(features):

    mask = ~np.isnan(features)

    return mask.astype(float)