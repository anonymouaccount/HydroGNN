import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

from config import (
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    DROPOUT,
    NUM_HEADS
)


# =========================
# FULL MODEL
# =========================
class FullModel(nn.Module):

    def __init__(self):

        super().__init__()

        # =========================
        # GAT ENCODER
        # =========================

        # First GAT layer
        self.gat1 = GATConv(
            in_channels=INPUT_DIM,
            out_channels=HIDDEN_DIM,
            heads=NUM_HEADS,
            dropout=DROPOUT,
            concat=True
        )

        # Second GAT layer
        self.gat2 = GATConv(
            in_channels=HIDDEN_DIM * NUM_HEADS,
            out_channels=OUTPUT_DIM,
            heads=1,
            dropout=DROPOUT,
            concat=False
        )

        # =========================
        # NODE RECONSTRUCTION HEAD
        # =========================
        self.reconstruction_head = nn.Sequential(

            nn.Linear(
                OUTPUT_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                HIDDEN_DIM,
                INPUT_DIM
            )
        )

        # =========================
        # EDGE DECODER (MLP)
        # =========================
        self.edge_decoder = nn.Sequential(

            nn.Linear(
                OUTPUT_DIM * 2,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM // 2
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM // 2,
                1
            )
        )

    # =========================
    # FORWARD
    # =========================
    def forward(
        self,
        x,
        edge_index
    ):

        # -------------------------
        # GAT Layer 1
        # -------------------------
        h = self.gat1(
            x,
            edge_index
        )

        h = F.elu(h)

        h = F.dropout(
            h,
            p=DROPOUT,
            training=self.training
        )

        # -------------------------
        # GAT Layer 2
        # -------------------------
        embeddings = self.gat2(
            h,
            edge_index
        )

        # -------------------------
        # Reconstruction
        # -------------------------
        x_reconstructed = (
            self.reconstruction_head(
                embeddings
            )
        )

        return (
            x_reconstructed,
            embeddings
        )

    # =========================
    # EDGE PREDICTION
    # =========================
    def predict_edges(
        self,
        embeddings,
        src,
        tgt
    ):

        # Source embeddings
        h_src = embeddings[src]

        # Target embeddings
        h_tgt = embeddings[tgt]

        # Concatenate
        edge_features = torch.cat(
            [h_src, h_tgt],
            dim=1
        )

        # Edge score
        scores = self.edge_decoder(
            edge_features
        )

        return scores.squeeze()