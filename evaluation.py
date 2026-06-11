import torch
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from config import DEVICE


# =========================
# MAIN EVALUATION
# =========================
def evaluate_model(
    model,
    graph_data
):

    print("   → Running evaluation...")

    model.eval()

    x = graph_data["x"].to(DEVICE)

    edge_index = graph_data[
        "train_edge_index"
    ].to(DEVICE)

    mask = graph_data["mask"].to(DEVICE)

    with torch.no_grad():

        # -------------------------
        # FORWARD
        # -------------------------
        x_pred, embeddings = model(
            x,
            edge_index
        )

    # =========================
    # RECONSTRUCTION METRICS
    # =========================
    rmse = compute_rmse(
        x_pred,
        x,
        mask
    )

    mae = compute_mae(
        x_pred,
        x,
        mask
    )

    anomaly_score = compute_anomaly_score(
        x_pred,
        x
    )

    connectivity = compute_connectivity(
        graph_data
    )

    results = {

        "RMSE": rmse,
        "MAE": mae,
        "AnomalyScore": anomaly_score,
        "Connectivity": connectivity
    }

    return results


# =========================
# EDGE EVALUATION
# =========================
def evaluate_edges(
    model,
    graph_data
):

    print(
        "   → Evaluating edge prediction..."
    )

    model.eval()

    x = graph_data["x"].to(DEVICE)

    train_edge_index = graph_data[
        "train_edge_index"
    ].to(DEVICE)

    test_edge_index = graph_data[
        "test_edge_index"
    ].to(DEVICE)

    with torch.no_grad():

        # -------------------------
        # EMBEDDINGS
        # -------------------------
        _, embeddings = model(
            x,
            train_edge_index
        )

    # =========================
    # POSITIVE TEST EDGES
    # =========================
    pos_src = test_edge_index[0]

    pos_tgt = test_edge_index[1]

    pos_scores = model.predict_edges(
        embeddings,
        pos_src,
        pos_tgt
    )

    pos_probs = torch.sigmoid(
        pos_scores
    )

    pos_labels = torch.ones(
        pos_probs.shape[0]
    )

    # =========================
    # NEGATIVE EDGES
    # =========================
    num_nodes = x.shape[0]

    neg_src = torch.randint(
        0,
        num_nodes,
        pos_src.shape,
        device=DEVICE
    )

    neg_tgt = torch.randint(
        0,
        num_nodes,
        pos_tgt.shape,
        device=DEVICE
    )

    neg_scores = model.predict_edges(
        embeddings,
        neg_src,
        neg_tgt
    )

    neg_probs = torch.sigmoid(
        neg_scores
    )

    neg_labels = torch.zeros(
        neg_probs.shape[0]
    )

    # =========================
    # COMBINE
    # =========================
    y_scores = torch.cat(
    [pos_probs, neg_probs]
    ).detach().cpu().numpy()

    y_true = torch.cat(
    [pos_labels, neg_labels]
    ).detach().cpu().numpy()

    # Binary prediction
    y_pred = (
        y_scores > 0.5
    ).astype(int)

    # =========================
    # METRICS
    # =========================
    auc = roc_auc_score(
        y_true,
        y_scores
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    return {

        "Edge_AUC": float(auc),

        "Edge_F1": float(f1),

        "Edge_Precision": float(
            precision
        ),

        "Edge_Recall": float(
            recall
        ),

        "Topology_Accuracy": float(
            accuracy
        )
    }


# =========================
# RMSE
# =========================
def compute_rmse(
    x_pred,
    x_true,
    mask
):

    diff = (
        (x_pred - x_true) ** 2
    ) * mask

    mse = diff.sum() / (
        mask.sum() + 1e-8
    )

    rmse = torch.sqrt(
        mse
    ).item()

    return rmse


# =========================
# MAE
# =========================
def compute_mae(
    x_pred,
    x_true,
    mask
):

    diff = torch.abs(
        x_pred - x_true
    ) * mask

    mae = diff.sum() / (
        mask.sum() + 1e-8
    )

    return mae.item()


# =========================
# ANOMALY SCORE
# =========================
def compute_anomaly_score(
    x_pred,
    x_true,
    threshold=0.5
):

    error = torch.abs(
        x_pred - x_true
    )

    anomalies = (
        error > threshold
    ).sum()

    return int(
        anomalies.item()
    )


# =========================
# CONNECTIVITY
# =========================
def compute_connectivity(
    graph_data
):

    edge_index = graph_data[
        "edge_index"
    ]

    num_nodes = graph_data[
        "x"
    ].shape[0]

    connected_nodes = torch.unique(
        edge_index
    )

    connectivity = (
        connected_nodes.shape[0]
        / num_nodes
    )

    return float(connectivity)