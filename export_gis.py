import os
import numpy as np
import geopandas as gpd

from shapely.geometry import (
    Point,
    LineString
)

from config import (
    OUTPUT_PATH,
    NODE_ID_COL
)


# =========================
# DETECT COORDINATE COLUMNS
# =========================
def detect_coordinate_columns(df):

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
        for c in df.columns
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

    return x_col, y_col


# =========================
# EXPORT NODE SHAPEFILE
# =========================
def export_nodes_shp(df_nodes):

    print("   → Exporting node shapefile...")

    df = df_nodes.copy()

    x_col, y_col = detect_coordinate_columns(df)

    # Convert safely
    df[x_col] = pd_to_numeric_safe(
        df[x_col]
    )

    df[y_col] = pd_to_numeric_safe(
        df[y_col]
    )

    # Remove invalid coords
    df = df.dropna(
        subset=[x_col, y_col]
    )

    geometry = [
        Point(xy)
        for xy in zip(
            df[x_col],
            df[y_col]
        )
    ]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:2154"
    )

    save_path = os.path.join(
        OUTPUT_PATH,
        "clean_nodes.shp"
    )

    gdf.to_file(save_path)

    print(
        f" Node shapefile saved: {save_path}"
    )


# =========================
# EXPORT ARC SHAPEFILE
# =========================
def export_arcs_shp(
    graph_data,
    df_nodes
):

    print("   → Exporting arc shapefile...")

    df = df_nodes.copy()

    x_col, y_col = detect_coordinate_columns(df)

    df[x_col] = pd_to_numeric_safe(
        df[x_col]
    )

    df[y_col] = pd_to_numeric_safe(
        df[y_col]
    )

    df = df.dropna(
        subset=[x_col, y_col]
    ).reset_index(drop=True)

    # =========================
    # CREATE NODE COORD MAP
    # =========================
    node_coords = {}

    for idx, row in df.iterrows():

        node_coords[idx] = (
            row[x_col],
            row[y_col]
        )

    # =========================
    # CREATE LINES
    # =========================
    edge_index = graph_data[
        "edge_index"
    ].cpu().numpy()

    lines = []

    valid_edges = []

    for src, tgt in edge_index.T:

        if (
            src in node_coords
            and tgt in node_coords
        ):

            p1 = node_coords[src]
            p2 = node_coords[tgt]

            line = LineString([
                (p1[0], p1[1]),
                (p2[0], p2[1])
            ])

            lines.append(line)

            valid_edges.append({
                "source": int(src),
                "target": int(tgt)
            })

    gdf = gpd.GeoDataFrame(
        valid_edges,
        geometry=lines,
        crs="EPSG:2154"
    )

    save_path = os.path.join(
        OUTPUT_PATH,
        "clean_arcs.shp"
    )

    gdf.to_file(save_path)

    print(
        f" Arc shapefile saved: {save_path}"
    )


# =========================
# EXPORT BASIN SHAPEFILE
# =========================
def export_basins_shp(df_basins):

    print("   → Exporting basin shapefile...")

    df = df_basins.copy()

    try:

        x_col, y_col = detect_coordinate_columns(df)

        df[x_col] = pd_to_numeric_safe(
            df[x_col]
        )

        df[y_col] = pd_to_numeric_safe(
            df[y_col]
        )

        df = df.dropna(
            subset=[x_col, y_col]
        )

        geometry = [
            Point(xy)
            for xy in zip(
                df[x_col],
                df[y_col]
            )
        ]

        gdf = gpd.GeoDataFrame(
            df,
            geometry=geometry,
            crs="EPSG:2154"
        )

    except:

        # Fallback without geometry
        gdf = gpd.GeoDataFrame(df)

    save_path = os.path.join(
        OUTPUT_PATH,
        "clean_basins.shp"
    )

    gdf.to_file(save_path)

    print(
        f" Basin shapefile saved: {save_path}"
    )


# =========================
# SAFE NUMERIC CONVERSION
# =========================
def pd_to_numeric_safe(series):

    return (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(
            lambda s: gpd.pd.to_numeric(
                s,
                errors="coerce"
            )
        )
    )