import os
import torch
import matplotlib.pyplot as plt

from config import (
    DATA_PATH,
    OUTPUT_PATH,
    DEVICE
)

from visualization import (
    plot_training_curve,
    plot_graph_topology,
    plot_embeddings,
    plot_anomaly_map,
    plot_edge_recovery
)

from preprocessing import (
    load_data,
    clean_data
)

from graph_builder import (
    build_graph
)

from train import (
    train_model
)

from evaluation import (
    evaluate_model,
    evaluate_edges
)

from export_results import (
    export_clean_nodes,
    export_clean_arcs,
    export_clean_basins
)

from export_gis import (
    export_nodes_shp,
    export_arcs_shp,
    export_basins_shp
)

from utils import set_seed

set_seed(42)


# =========================
# MAIN PIPELINE
# =========================
def main():

    print(
        " Starting Drainage Data Quality Pipeline...\n"
    )

    # =========================
    # 1. LOAD DATA
    # =========================
    print(" Loading data...")

    nodes_path = os.path.join(
        DATA_PATH,
        "MMM_MMM_ReseauPluvialNoeuds.csv"
    )

    arcs_path = os.path.join(
        DATA_PATH,
        "MMM_MMM_ReseauPluvialLineaire.csv"
    )

    basins_path = os.path.join(
        DATA_PATH,
        "MMM_MMM_ReseauPluvialBassins.csv"
    )

    df_nodes, df_arcs, df_basins = load_data(
        nodes_path,
        arcs_path,
        basins_path
    )

    print(
        " Data loaded successfully\n"
    )

    # =========================
    # 2. PREPROCESSING
    # =========================
    print(" Cleaning data...")

    df_nodes, df_arcs, df_basins = clean_data(
        df_nodes,
        df_arcs,
        df_basins
    )

    print(
        " Data cleaning completed\n"
    )

    # =========================
    # 3. GRAPH CONSTRUCTION
    # =========================
    print(" Building graph...")

    graph_data = build_graph(
        df_nodes,
        df_arcs,
        df_basins
    )

    print(
        " Graph constructed\n"
    )

    # =========================
    # GRAPH STATISTICS
    # =========================
    print(" Graph statistics...")

    num_nodes = graph_data["x"].shape[0]

    num_edges = graph_data[
        "edge_index"
    ].shape[1]

    print(
        f" Number of nodes: {num_nodes}"
    )

    print(
        f" Number of edges: {num_edges}\n"
    )

    # =========================
    # 4. TRAIN MODEL
    # =========================
    print(" Training GNN model...")

    model, training_logs = train_model(
        graph_data
    )

    print(
        " Training completed\n"
    )

    # =========================
    # 5. EVALUATION
    # =========================
    print(" Evaluating model...")

    # -------------------------
    # Reconstruction metrics
    # -------------------------
    results = evaluate_model(
        model,
        graph_data
    )

    # -------------------------
    # Edge prediction metrics
    # -------------------------
    edge_metrics = evaluate_edges(
        model,
        graph_data
    )

    # Merge dictionaries
    results.update(edge_metrics)

    print(
        " Evaluation completed\n"
    )

    # =========================
    # 6. SAVE RESULTS
    # =========================
    print(" Saving results...")

    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True
    )

    # -------------------------
    # SAVE METRICS
    # -------------------------
    results_path = os.path.join(
        OUTPUT_PATH,
        "results.txt"
    )

    with open(results_path, "w") as f:

        for key, value in results.items():

            f.write(
                f"{key}: {value}\n"
            )

    print(
        f" Results saved to: {results_path}"
    )

    # =========================
    # SAVE MODEL
    # =========================
    model_path = os.path.join(
        OUTPUT_PATH,
        "gnn_model.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f" Model saved to: {model_path}"
    )

    # =========================
    # SAVE LOSS CURVE
    # =========================
    loss_plot_path = os.path.join(
        OUTPUT_PATH,
        "loss_curve.png"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        training_logs
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Training Loss Curve"
    )

    plt.grid(True)

    plt.savefig(
        loss_plot_path
    )

    plt.close()

    print(
        f" Loss curve saved to: {loss_plot_path}"
    )

    print(" Generating visualizations...")

    plot_training_curve(
    training_logs
    )

    plot_graph_topology(
    graph_data
    )

    plot_embeddings(
    model,
    graph_data,
    DEVICE
    )

    plot_anomaly_map(
    model,
    graph_data,
    DEVICE
    )

    plot_edge_recovery(
    graph_data
    )

    print(" Visualizations completed\n")

    # =========================
    # EXPORT CLEAN DATASETS
    # =========================
    print(" Exporting reconstructed datasets...")

    export_clean_nodes(
        model,
        graph_data,
        df_nodes,
        DEVICE
    )

    export_clean_arcs(
        graph_data,
        df_arcs
    )

    export_clean_basins(
        df_basins
    )

    print(" Dataset export completed\n")

# =========================
# EXPORT GIS FILES
# =========================
    print(" Exporting GIS shapefiles...")

    export_nodes_shp(
    df_nodes
    )

    export_arcs_shp(
    graph_data,
    df_nodes
    )

    export_basins_shp(
    df_basins
    )

    print(" GIS export completed\n")

    print(
        "\n Pipeline completed successfully!"
    )


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":

    main()