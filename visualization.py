import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.decomposition import PCA

from config import OUTPUT_PATH


# =========================
# TRAINING CURVE
# =========================
def plot_training_curve(training_logs):

    plt.figure(figsize=(8, 5))

    plt.plot(training_logs)

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss Curve")

    plt.grid(True)

    save_path = os.path.join(
        OUTPUT_PATH,
        "training_curve.png"
    )

    plt.savefig(save_path)

    plt.close()

    print(f" Training curve saved: {save_path}")


# =========================
# GRAPH TOPOLOGY
# =========================
def plot_graph_topology(graph_data):

    edge_index = graph_data[
        "edge_index"
    ].cpu().numpy()

    G = nx.Graph()

    edges = edge_index.T

    G.add_edges_from(edges)

    plt.figure(figsize=(10, 10))

    pos = nx.spring_layout(
        G,
        seed=42
    )

    nx.draw(
        G,
        pos,
        node_size=2,
        width=0.2,
        with_labels=False
    )

    plt.title(
        "Reconstructed Drainage Topology"
    )

    save_path = os.path.join(
        OUTPUT_PATH,
        "graph_topology.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f" Graph topology saved: {save_path}")


# =========================
# LATENT EMBEDDINGS
# =========================
def plot_embeddings(
    model,
    graph_data,
    device
):

    model.eval()

    x = graph_data["x"].to(device)

    edge_index = graph_data[
        "train_edge_index"
    ].to(device)

    with torch.no_grad():

        _, embeddings = model(
            x,
            edge_index
        )

    embeddings = embeddings.cpu().numpy()

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(
        embeddings
    )

    plt.figure(figsize=(10, 8))

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=3,
        alpha=0.7
    )

    plt.title(
        "Latent Node Embeddings"
    )

    plt.xlabel("PCA 1")

    plt.ylabel("PCA 2")

    save_path = os.path.join(
        OUTPUT_PATH,
        "latent_embeddings.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f" Embeddings saved: {save_path}")


# =========================
# ANOMALY MAP
# =========================
def plot_anomaly_map(
    model,
    graph_data,
    device,
    threshold=0.1
):

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

    errors = torch.mean(
        torch.abs(x_pred - x),
        dim=1
    )

    errors = errors.cpu().numpy()

    anomalies = errors > threshold

    plt.figure(figsize=(10, 6))

    plt.hist(
        errors,
        bins=50
    )

    plt.axvline(
        threshold,
        linestyle="--"
    )

    plt.title(
        "Anomaly Distribution"
    )

    plt.xlabel(
        "Reconstruction Error"
    )

    plt.ylabel(
        "Frequency"
    )

    save_path = os.path.join(
        OUTPUT_PATH,
        "anomaly_distribution.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f" Anomaly map saved: {save_path}")

    print(f" Detected anomalies: {anomalies.sum()}")


# =========================
# EDGE RECOVERY
# =========================
def plot_edge_recovery(graph_data):

    train_edges = graph_data[
        "train_edge_index"
    ].shape[1]

    test_edges = graph_data[
        "test_edge_index"
    ].shape[1]

    total_edges = graph_data[
        "edge_index"
    ].shape[1]

    labels = [
        "Train",
        "Test",
        "Total"
    ]

    values = [
        train_edges,
        test_edges,
        total_edges
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(labels, values)

    plt.title(
        "Recovered Edge Statistics"
    )

    plt.ylabel(
        "Number of Edges"
    )

    save_path = os.path.join(
        OUTPUT_PATH,
        "edge_recovery.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f" Edge recovery figure saved: {save_path}")