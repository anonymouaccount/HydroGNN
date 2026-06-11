import os
import torch
import pandas as pd
import numpy as np

from config import OUTPUT_PATH


# =========================
# EXPORT CLEAN NODES
# =========================
def export_clean_nodes(
    model,
    graph_data,
    df_nodes,
    device
):

    print("   → Exporting clean nodes...")

    model.eval()

    x = graph_data["x"].to(device)

    edge_index = graph_data[
        "train_edge_index"
    ].to(device)

    with torch.no_grad():

        x_pred, _ = model(
            x,
            edge_index
        )

    reconstructed = (
        x_pred.cpu().numpy()
    )

    clean_nodes = df_nodes.copy()

    numeric_cols = clean_nodes.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    max_cols = min(
        len(numeric_cols),
        reconstructed.shape[1]
    )

    # =========================
    # Replace reconstructed values
    # =========================
    for i in range(max_cols):

        original_col = numeric_cols[i]

        clean_nodes[
            f"reconstructed_{original_col}"
        ] = reconstructed[:, i]

    # =========================
    # Anomaly score
    # =========================
    original_tensor = (
        x.cpu().numpy()
    )

    reconstruction_error = np.mean(
        np.abs(
            reconstructed
            - original_tensor
        ),
        axis=1
    )

    clean_nodes[
        "anomaly_score"
    ] = reconstruction_error

    clean_nodes[
        "is_anomaly"
    ] = (
        reconstruction_error > 0.1
    ).astype(int)

    # =========================
    # Save
    # =========================
    save_path = os.path.join(
        OUTPUT_PATH,
        "clean_nodes.csv"
    )

    clean_nodes.to_csv(
        save_path,
        index=False
    )

    print(
        f" Clean nodes saved: {save_path}"
    )


# =========================
# EXPORT CLEAN ARCS
# =========================
def export_clean_arcs(
    graph_data,
    df_arcs
):

    print("   → Exporting clean arcs...")

    clean_arcs = df_arcs.copy()

    edge_index = graph_data[
        "edge_index"
    ].cpu().numpy()

    sources = edge_index[0]
    targets = edge_index[1]

    reconstructed_edges = pd.DataFrame({
        "source_node": sources,
        "target_node": targets
    })

    reconstructed_edges[
        "recovered_edge"
    ] = 1

    save_path = os.path.join(
        OUTPUT_PATH,
        "clean_arcs.csv"
    )

    reconstructed_edges.to_csv(
        save_path,
        index=False
    )

    print(
        f" Clean arcs saved: {save_path}"
    )


# =========================
# EXPORT CLEAN BASINS
# =========================
def export_clean_basins(
    df_basins
):

    print("   → Exporting clean basins...")

    clean_basins = df_basins.copy()

    save_path = os.path.join(
        OUTPUT_PATH,
        "clean_basins.csv"
    )

    clean_basins.to_csv(
        save_path,
        index=False
    )

    print(
        f" Clean basins saved: {save_path}"
    )