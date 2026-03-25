#!/usr/bin/env python3
"""
tree_CO2_seq_and_map.py

Usage
-----
python tree_CO2_seq_and_map.py \
    --run-dir /home/fss6k/datasets/communimap_trees_TammyCampos_20260324_104150 \
    --survey-file "/mnt/c/Users/fss6k/OneDrive - University of Glasgow/Desktop/GALLANT_WS3_documents_data/CommuniMap/March_26_data/communiMap data March 26.xlsx" \
    --outdir /home/fss6k/datasets/communimap_trees_TammyCampos_20260324_104150 \
    --map-name tree_sequestration_map.html \
    --csv-name tree_carbon_estimates.csv

What it does
------------
1. Loads CommuniMap survey data and image-level predictions
2. Aggregates image-level predictions to tree ID
3. Converts height/diameter classes to midpoint values with uncertainty
4. Computes carbon stock using the provided Excel formula:
      IF(height < 28, 0.0577*height^2*diameter, 0.0346*height^2*diameter) * 0.25
   where:
      - height is in meters
      - diameter is in cm
5. Computes annual C sequestration and CO2e sequestration using:
      sequestration = r * carbon_stock
6. Propagates uncertainty into C and CO2e sequestration
7. Converts CO2e into kettle boils using:
      kettle_boils = CO2e_kg / 0.018
8. Saves enriched CSV and Folium HTML map
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

import branca.colormap as cm
from branca.element import Element
import folium
import numpy as np
import pandas as pd


# =========================================================
# User-adjustable assumptions
# =========================================================

# Annual sequestration model: dC/dt = r * C
SEQ_RATE_MEAN = 0.02   # yr^-1
SEQ_RATE_UNC = 0.01    # yr^-1

# Kettle conversion
KG_CO2E_PER_KETTLE_BOIL = 0.018

# Height classes -> (midpoint, uncertainty)
# Include both "<5" and "0-5" for compatibility
HEIGHT_CLASS_MAP = {
    "0-5":  (2.5, 2.5),
    "5-10": (7.5, 2.5),
    "10-15": (12.5, 2.5),
    "15+":  (17.5, 2.5),
}

# Diameter classes -> (midpoint, uncertainty), in cm
DIAMETER_CLASS_MAP = {
    "0-20":  (10.0, 5.0),
    "20-40": (30.0, 5.0),
    "40-60": (50.0, 5.0),
    "60+":   (70.0, 5.0),
}


# =========================================================
# IO helpers
# =========================================================

def sniff_delimiter(path: Path, default: str = ";") -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except Exception:
        return default


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        delim = sniff_delimiter(path)
        return pd.read_csv(path, delimiter=delim)
    raise ValueError(f"Unsupported file type: {path.suffix}")


# =========================================================
# Species cleaning
# =========================================================

def clean_species_name(x) -> Optional[str]:
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()
    x = x.encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"\(.*?\)", "", x)
    x = x.replace("_", " ").replace("-", " / ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def canonical_species_group(s: Optional[str]) -> str:
    if pd.isna(s):
        return "unknown"

    s = str(s).lower().strip()

    if s in {"", "unknown", "type of maple", "tree"}:
        return "unknown"
    if "fern" in s or "rhododendron" in s:
        return "unknown"

    if "beech" in s:
        return "beech"
    if "oak" in s:
        return "oak"
    if "ash" in s:
        return "ash"
    if "lime" in s:
        return "lime"
    if "sycamore" in s:
        return "sycamore"
    if "maple" in s:
        return "maple"
    if "birch" in s:
        return "birch"
    if "hawthorn" in s or "hawthorne" in s or "crataegus" in s:
        return "hawthorn"
    if "holly" in s or "ilex" in s:
        return "holly"
    if "rowan" in s:
        return "rowan"
    if "alder" in s or "alnus" in s:
        return "alder"
    if "hazel" in s:
        return "hazel"
    if "hornbeam" in s:
        return "hornbeam"
    if "whitebeam" in s:
        return "whitebeam"
    if "plane" in s or "platanus" in s:
        return "plane"
    if "chestnut" in s:
        return "chestnut"
    if "cherry" in s or "plum" in s or "apple" in s:
        return "cherry_plum"
    if "poplar" in s or "willow" in s:
        return "poplar_willow"
    if "walnut" in s:
        return "walnut"
    if "elm" in s:
        return "elm"

    if any(k in s for k in ["magnolia", "catalpa", "cotoneaster", "eucalyptus", "ficus", "avocado", "orange", "box"]):
        return "ornamental_broadleaf"

    if "scots pine" in s:
        return "pine"
    if "pine" in s or "pinus" in s:
        return "pine"
    if any(k in s for k in ["cedar", "cypress", "fir", "spruce", "larch", "yew"]):
        return "conifer_other"

    if "palm" in s:
        return "ornamental_other"

    return "unknown"


# =========================================================
# Label -> numeric with uncertainty
# =========================================================

def class_to_value_unc(label: object, mapping: dict[str, tuple[float, float]]) -> tuple[float, float]:
    if pd.isna(label):
        return np.nan, np.nan

    label = str(label).strip()
    if label in mapping:
        return mapping[label]

    return np.nan, np.nan


def aggregate_class_list(labels: object, mapping: dict[str, tuple[float, float]]) -> tuple[float, float, int]:
    """
    Convert a list of class labels to:
      mean midpoint,
      combined uncertainty,
      count of valid labels

    Combined uncertainty:
      sqrt( mean(class_unc^2) + sample_variance_across_predictions )
    """
    if not isinstance(labels, list) or len(labels) == 0:
        return np.nan, np.nan, 0

    vals = []
    uncs = []

    for x in labels:
        v, u = class_to_value_unc(x, mapping)
        if pd.notna(v):
            vals.append(float(v))
            uncs.append(float(u))

    n = len(vals)
    if n == 0:
        return np.nan, np.nan, 0

    mean_val = float(np.mean(vals))
    disagreement_var = float(np.var(vals, ddof=1)) if n > 1 else 0.0
    class_unc2 = float(np.mean(np.square(uncs)))
    combined_unc = float(np.sqrt(class_unc2)) #no disagreement for now 

    return mean_val, combined_unc, n


# =========================================================
# Carbon stock + uncertainty
# =========================================================

def carbon_stock_from_height_diameter(height_m: float, diameter_cm: float) -> float:
    """
    Excel formula:
      =IF(E<28, 0.0577*E^2*D, 0.0346*E^2*D) * 0.25

    height_m: meters
    diameter_cm: centimeters
    """
    if pd.isna(height_m) or pd.isna(diameter_cm):
        return np.nan

    if height_m < 28:
        return 0.0577 * (height_m ** 2) * diameter_cm * 0.25
    return 0.0346 * (height_m ** 2) * diameter_cm * 0.25


def carbon_stock_uncertainty(
    height_m: float,
    height_unc_m: float,
    diameter_cm: float,
    diameter_unc_cm: float,
    carbon_stock_kg: float
) -> float:
    """
    For C = k * H^2 * D

    relative uncertainty:
      (u_C / C)^2 = (2 u_H / H)^2 + (u_D / D)^2
    """
    if any(pd.isna(x) for x in [height_m, height_unc_m, diameter_cm, diameter_unc_cm, carbon_stock_kg]):
        return np.nan

    if height_m <= 0 or diameter_cm <= 0 or carbon_stock_kg <= 0:
        return np.nan

    rel2 = (
        (2.0 * height_unc_m / height_m) ** 2
        + (diameter_unc_cm / diameter_cm) ** 2
    )
    return float(carbon_stock_kg * np.sqrt(rel2))


# =========================================================
# Misc helpers
# =========================================================

def first_image(x):
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    if isinstance(x, str):
        return x
    return None


def normalise_species_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SPECIES"] = df["TREE_TYPE"]

    other_mask = (
        df["TREE_TYPE"].notna()
        & df["TREE_TYPE"].astype(str).str.lower().eq("other")
    )
    df.loc[other_mask, "SPECIES"] = df.loc[other_mask, "OTHER_TREE"]

    df["SPECIES"] = df["SPECIES"].astype("object")
    df["SPECIES"] = df["SPECIES"].where(df["SPECIES"].notna(), np.nan)

    replacements = {
        "copper beech": "beech",
        "whitebeam": "whitebeam",
        "plum tree": "plum",
    }
    df["SPECIES"] = df["SPECIES"].replace(replacements)
    return df


# =========================================================
# Main pipeline
# =========================================================

def build_tree_table(run_dir: Path, survey_file: Path, pred_file_rel: str) -> pd.DataFrame:
    pred_path = run_dir / pred_file_rel

    df_pred = load_table(pred_path)
    df_survey = load_table(survey_file)

    print(f"Predictions shape: {df_pred.shape}")
    print(f"Survey shape:      {df_survey.shape}")

    df_survey = normalise_species_column(df_survey)

    if "TREE" in df_survey.columns:
        df_survey = df_survey[df_survey["TREE"].astype(str).str.lower().eq("yes")].copy()

    pred_agg = (
        df_pred
        .groupby("ID", dropna=False)
        .agg({
            "HEIGHT_CLASS_FINAL_STR": lambda x: list(x.dropna()),
            "DIAMETER_CLASS_FINAL_STR": lambda x: list(x.dropna()),
            "MEDIA_SRC": lambda x: list(x.dropna()),
        })
        .reset_index()
    )

    df = df_survey.merge(pred_agg, on="ID", how="left")

    df["SPECIES_CLEAN"] = df["SPECIES"].apply(clean_species_name)
    df["SPECIES_GROUP"] = df["SPECIES_CLEAN"].apply(canonical_species_group)

    height_stats = df["HEIGHT_CLASS_FINAL_STR"].apply(
        lambda x: aggregate_class_list(x, HEIGHT_CLASS_MAP)
    )
    df[["HEIGHT_M", "HEIGHT_UNC_M", "HEIGHT_N_PRED"]] = pd.DataFrame(
        height_stats.tolist(), index=df.index
    )

    diam_stats = df["DIAMETER_CLASS_FINAL_STR"].apply(
        lambda x: aggregate_class_list(x, DIAMETER_CLASS_MAP)
    )
    df[["DIAMETER_CM", "DIAMETER_UNC_CM", "DIAMETER_N_PRED"]] = pd.DataFrame(
        diam_stats.tolist(), index=df.index
    )

    valid = df["HEIGHT_M"].notna() & df["DIAMETER_CM"].notna()

    df["CARBON_STOCK_KG"] = np.nan
    df.loc[valid, "CARBON_STOCK_KG"] = df.loc[valid].apply(
        lambda r: carbon_stock_from_height_diameter(r["HEIGHT_M"], r["DIAMETER_CM"]),
        axis=1
    )

    df["CARBON_STOCK_UNC_KG"] = np.nan
    df.loc[valid, "CARBON_STOCK_UNC_KG"] = df.loc[valid].apply(
        lambda r: carbon_stock_uncertainty(
            r["HEIGHT_M"],
            r["HEIGHT_UNC_M"],
            r["DIAMETER_CM"],
            r["DIAMETER_UNC_CM"],
            r["CARBON_STOCK_KG"],
        ),
        axis=1
    )

    df["SEQ_RATE_YR"] = SEQ_RATE_MEAN
    df["SEQ_RATE_UNC_YR"] = SEQ_RATE_UNC

    df["CARBON_SEQUESTERED_KG_YR"] = df["SEQ_RATE_YR"] * df["CARBON_STOCK_KG"]
    df["CARBON_SEQUESTERED_UNC_KG_YR"] = np.sqrt(
        (df["CARBON_STOCK_KG"] * df["SEQ_RATE_UNC_YR"]) ** 2
        + (df["SEQ_RATE_YR"] * df["CARBON_STOCK_UNC_KG"]) ** 2
    )

    co2_factor = 44.0 / 12.0
    df["CO2_SEQUESTERED_KG_YR"] = df["CARBON_SEQUESTERED_KG_YR"] * co2_factor
    df["CO2_SEQUESTERED_UNC_KG_YR"] = df["CARBON_SEQUESTERED_UNC_KG_YR"] * co2_factor

    # Kettle boils
    df["KETTLE_BOILS_PER_YEAR"] = df["CO2_SEQUESTERED_KG_YR"] / KG_CO2E_PER_KETTLE_BOIL
    df["KETTLE_BOILS_UNC_PER_YEAR"] = df["CO2_SEQUESTERED_UNC_KG_YR"] / KG_CO2E_PER_KETTLE_BOIL

    df["IMAGE_URL"] = df["MEDIA_SRC"].apply(first_image)

    print("Non-null carbon stock rows:", int(df["CARBON_STOCK_KG"].notna().sum()))
    print("Non-null C sequestration rows:", int(df["CARBON_SEQUESTERED_KG_YR"].notna().sum()))
    print("Non-null CO2 sequestration rows:", int(df["CO2_SEQUESTERED_KG_YR"].notna().sum()))

    return df


# =========================================================
# Mapping
# =========================================================

def make_popup_html(row, colormap) -> str:
    c_seq = row["CARBON_SEQUESTERED_KG_YR"]
    c_seq_unc = row.get("CARBON_SEQUESTERED_UNC_KG_YR", np.nan)
    co2_seq = row.get("CO2_SEQUESTERED_KG_YR", np.nan)
    co2_seq_unc = row.get("CO2_SEQUESTERED_UNC_KG_YR", np.nan)
    kettles = row.get("KETTLE_BOILS_PER_YEAR", np.nan)
    kettles_unc = row.get("KETTLE_BOILS_UNC_PER_YEAR", np.nan)

    color = colormap(c_seq)

    species = row.get("SPECIES_GROUP", "unknown")
    tree_id = row.get("ID", "NA")
    img_url = row.get("IMAGE_URL", None)

    img_html = ""
    if pd.notna(img_url) and str(img_url).startswith("http"):
        img_html = f"""
        <div style="margin-top:10px; text-align:center;">
            <img src="{img_url}" style="width:300px; max-width:100%; border-radius:12px; border:3px solid {color};">
        </div>
        """

    html = f"""
    <div style="width:360px; font-family:Arial, sans-serif;">
        <div style="font-size:28px; font-weight:900; color:{color}; text-align:center; line-height:1.1;">
            {c_seq:.3f} ± {c_seq_unc:.3f}
        </div>

        <div style="font-size:15px; font-weight:700; color:{color}; margin-bottom:10px; text-align:center;">
            kg C / year
        </div>

        <div style="font-size:16px; font-weight:800; margin-bottom:8px; text-align:center;">
            {co2_seq:.3f} ± {co2_seq_unc:.3f} kg CO₂e / year
        </div>

        <div style="font-size:15px; margin-bottom:10px; text-align:center;">
            ≈ {kettles:,.1f} ± {kettles_unc:,.1f} kettle boils / year
        </div>

        <div style="font-size:14px; margin-bottom:4px;"><b>ID:</b> {tree_id}</div>
        <div style="font-size:14px; margin-bottom:4px;"><b>Species:</b> {species}</div>
        <div style="font-size:14px; margin-bottom:4px;"><b>Height:</b> {row.get("HEIGHT_M", np.nan):.1f} ± {row.get("HEIGHT_UNC_M", np.nan):.1f} m</div>
        <div style="font-size:14px; margin-bottom:4px;"><b>Diameter:</b> {row.get("DIAMETER_CM", np.nan):.1f} ± {row.get("DIAMETER_UNC_CM", np.nan):.1f} cm</div>

        <div style="font-size:12px; color:#444; margin-top:8px;">
            Lat: {row["LATITUDE"]:.5f}<br>
            Lon: {row["LONGITUDE"]:.5f}
        </div>

        {img_html}
    </div>
    """
    return html


def marker_radius(value: float, vmin: float, vmax: float) -> float:
    if pd.isna(value):
        return 5.0
    return max(5.0, min(14.0, 5.0 + 9.0 * ((value - vmin) / (vmax - vmin + 1e-9))))


def build_map(df: pd.DataFrame, out_html: Path) -> None:
    df_map = df.copy()

    df_map = df_map[
        df_map["LATITUDE"].notna()
        & df_map["LONGITUDE"].notna()
        & df_map["CARBON_SEQUESTERED_KG_YR"].notna()
    ].copy()

    print(f"Rows going into map: {len(df_map)}")

    if df_map.empty:
        print("No valid rows for map. Skipping map export.")
        return

    UNI_LAT = 55.8721
    UNI_LON = -4.2890

    m = folium.Map(
        location=[UNI_LAT, UNI_LON],
        zoom_start=12,
        tiles="CartoDB positron"
    )

    folium.Marker(
        location=[UNI_LAT, UNI_LON],
        popup="University of Glasgow",
        icon=folium.Icon(color="purple", icon="university", prefix="fa")
    ).add_to(m)

    vmin = float(df_map["CARBON_SEQUESTERED_KG_YR"].min())
    vmax = float(df_map["CARBON_SEQUESTERED_KG_YR"].max())
    if vmin == vmax:
        vmax = vmin + 1e-6

    colormap = cm.LinearColormap(
        colors=["green", "yellow", "orange", "red"],
        vmin=vmin,
        vmax=vmax
    )
    colormap.caption = "Carbon sequestered (kg C / year)"
    colormap.add_to(m)

    for _, row in df_map.iterrows():
        seq = float(row["CARBON_SEQUESTERED_KG_YR"])
        color = colormap(seq)
        popup_html = make_popup_html(row, colormap)

        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=marker_radius(seq, vmin, vmax),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(popup_html, max_width=390)
        ).add_to(m)

    total_c = round(df_map["CARBON_SEQUESTERED_KG_YR"].sum())
    total_c_unc = round(float(np.sqrt(np.square(df_map["CARBON_SEQUESTERED_UNC_KG_YR"].fillna(0)).sum())))

    total_co2 = round(df_map["CO2_SEQUESTERED_KG_YR"].sum())
    total_co2_unc = round(float(np.sqrt(np.square(df_map["CO2_SEQUESTERED_UNC_KG_YR"].fillna(0)).sum())))

    total_kettles = round(df_map["KETTLE_BOILS_PER_YEAR"].sum())
    total_kettles_unc = round(float(np.sqrt(np.square(df_map["KETTLE_BOILS_UNC_PER_YEAR"].fillna(0)).sum())))

    html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 320px;
        z-index: 9999;
        font-size: 15px;
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        border-left: 6px solid green;
        font-family: Arial, sans-serif;
    ">
        <div style="font-size:18px; font-weight:800; margin-bottom:10px;">
            Glasgow Carbon Capture from {len(df_map)} trees
        </div>

        <div style="margin-bottom:8px;">
            <b>Total C sequestration:</b><br>
            <span style="color:green; font-weight:800;">{total_c:,.0f} ± {total_c_unc:,.0f}</span> kg C / year
        </div>

        <div style="margin-bottom:8px;">
            <b>Total CO₂e sequestration:</b><br>
            <span style="color:#444; font-weight:800;">{total_co2:,.0f} ± {total_co2_unc:,.0f}</span> kg CO₂ / year
        </div>

        <div>
            <b>Kettle boils equivalent:</b><br>
            <span style="color:#444; font-weight:800;">{total_kettles:,.0f} ± {total_kettles_unc:,.0f}</span> kettle boils / year
        </div>
    </div>
    """

    m.get_root().html.add_child(Element(html))

    m.save(str(out_html))
    print(f"Saved map to: {out_html}")


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tree carbon pipeline from CommuniMap survey + image predictions.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Folder for the tree prediction run, e.g. /home/.../communimap_trees_xxx"
    )
    parser.add_argument(
        "--survey-file",
        type=Path,
        required=True,
        help="Path to March/other CommuniMap survey file (.xlsx or .csv)"
    )
    parser.add_argument(
        "--pred-file",
        type=str,
        default="manifests/tree_dataset_manifest_with_diameter_predictions.csv",
        help="Path relative to --run-dir for the image-level prediction manifest"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Default: <run-dir>/outputs"
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="tree_sequestration_map.html",
        help="Name of the output HTML map"
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="tree_carbon_estimates.csv",
        help="Name of the enriched output CSV"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outdir = args.outdir if args.outdir is not None else args.run_dir / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    df = build_tree_table(
        run_dir=args.run_dir,
        survey_file=args.survey_file,
        pred_file_rel=args.pred_file,
    )

    out_csv = outdir / args.csv_name
    out_map = outdir / args.map_name

    df.to_csv(out_csv, index=False)
    print(f"Saved enriched table to: {out_csv}")

    build_map(df, out_map)

    preview_cols = [
        "ID",
        "SPECIES",
        "SPECIES_GROUP",
        "HEIGHT_M", "HEIGHT_UNC_M",
        "DIAMETER_CM", "DIAMETER_UNC_CM",
        "CARBON_STOCK_KG", "CARBON_STOCK_UNC_KG",
        "SEQ_RATE_YR", "SEQ_RATE_UNC_YR",
        "CARBON_SEQUESTERED_KG_YR", "CARBON_SEQUESTERED_UNC_KG_YR",
        "CO2_SEQUESTERED_KG_YR", "CO2_SEQUESTERED_UNC_KG_YR",
        "KETTLE_BOILS_PER_YEAR", "KETTLE_BOILS_UNC_PER_YEAR",
        "IMAGE_URL",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nPreview:")
    print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()