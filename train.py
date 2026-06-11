import torch
import torch.nn.functional as F
import torch.optim as optim

from model import FullModel

from config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LAMBDA_REC,
    LAMBDA_LINK,
    LAMBDA_PHYS,
    DEVICE
)


# =========================
# MAIN TRAIN FUNCTION
# =========================
def train_model(graph_data):

    print("   → Initializing model...")

    model = FullModel().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    x = graph_data["x"].to(DEVICE)
    edge_index = graph_data["edge_index"].to(DEVICE)
    mask = graph_data["mask"].to(DEVICE)

    print("   → Starting training...\n")

    training_logs = []

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        # =========================
        # FORWARD
        # =========================
        x_pred, z = model(
            x,
            edge_index
        )

        # =========================
        # RECONSTRUCTION LOSS
        # =========================
        loss_rec = reconstruction_loss(
            x_pred,
            x,
            mask
        )

        # =========================
        # LINK PREDICTION LOSS
        # =========================
        loss_link = link_prediction_loss(
        model,
        z,
        edge_index
        )
        # =========================
        # PHYSICS LOSS
        # =========================
        loss_phys = physics_loss(
            x_pred
        )

        # =========================
        # TOTAL LOSS
        # =========================
        loss = (
            LAMBDA_REC * loss_rec
            + LAMBDA_LINK * loss_link
            + LAMBDA_PHYS * loss_phys
        )

        # =========================
        # SAFETY CHECK
        # =========================
        if torch.isnan(loss):

            print(" NaN loss detected.")

            break

        # =========================
        # BACKPROP
        # =========================
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        # =========================
        # LOGGING
        # =========================
        if epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Total: {loss.item():.4f} | "
                f"Rec: {loss_rec.item():.4f} | "
                f"Link: {loss_link.item():.4f} | "
                f"Phys: {loss_phys.item():.4f}"
            )

        training_logs.append(
            loss.item()
        )

    return model, training_logs


# =========================
# RECONSTRUCTION LOSS
# =========================
def reconstruction_loss(
    x_pred,
    x_true,
    mask
):

    diff = (x_pred - x_true) ** 2

    masked_diff = diff * mask

    loss = masked_diff.sum() / (
        mask.sum() + 1e-8
    )

    return loss


# =========================
# LINK PREDICTION LOSS
# =========================
def link_prediction_loss(
    model,
    embeddings,
    edge_index
):

    src = edge_index[0]

    tgt = edge_index[1]

    # =========================
    # POSITIVE EDGES
    # =========================
    pos_scores = model.predict_edges(
        embeddings,
        src,
        tgt
    )

    pos_labels = torch.ones_like(
        pos_scores
    )

    # =========================
    # NEGATIVE SAMPLING
    # =========================
    num_nodes = embeddings.shape[0]

    neg_src = torch.randint(
        0,
        num_nodes,
        src.shape,
        device=src.device
    )

    neg_tgt = torch.randint(
        0,
        num_nodes,
        tgt.shape,
        device=tgt.device
    )

    # =========================
    # NEGATIVE SCORES
    # =========================
    neg_scores = model.predict_edges(
        embeddings,
        neg_src,
        neg_tgt
    )

    neg_labels = torch.zeros_like(
        neg_scores
    )

    # =========================
    # COMBINE
    # =========================
    scores = torch.cat(
        [pos_scores, neg_scores],
        dim=0
    )

    labels = torch.cat(
        [pos_labels, neg_labels],
        dim=0
    )

    # =========================
    # BCE LOSS
    # =========================
    loss = F.binary_cross_entropy_with_logits(
        scores,
        labels
    )

    return loss


# =========================
# PHYSICS LOSS
# =========================
def physics_loss(x_pred):

    loss = 0.0

    # =========================
    # FEATURE INDICES
    # Adjust according to your
    # NODE_FEATURES order
    # =========================
    elevation_idx = 2
    depth_idx = 3
    slope_idx = 4

    # =========================
    # DEPTH CONSTRAINT
    # depth >= 0
    # =========================
    if x_pred.shape[1] > depth_idx:

        depth = x_pred[:, depth_idx]

        loss += torch.mean(
            F.relu(-depth)
        )

    # =========================
    # SLOPE CONSTRAINT
    # slope >= 0
    # =========================
    if x_pred.shape[1] > slope_idx:

        slope = x_pred[:, slope_idx]

        loss += torch.mean(
            F.relu(-slope)
        )

    # =========================
    # ELEVATION SMOOTHNESS
    # =========================
    if x_pred.shape[1] > elevation_idx:

        elevation = x_pred[:, elevation_idx]

        smoothness = torch.mean(
            torch.abs(
                elevation[1:]
                - elevation[:-1]
            )
        )

        loss += smoothness * 0.01

    return loss