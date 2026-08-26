"""
Exploratory data analysis, data quality checks, static charts, and an
interactive webmap for the Bengaluru bus stop audit dataset.

Run with:
    python eda_bus_stop_audit.py
"""

import json
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "bengaluru_bus_stop_audit_2026.csv"
BOUNDARY_PATH = BASE_DIR / "blr_boundary.geojson"
OUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUT_DIR / "charts"
REPORT_PATH = OUT_DIR / "data_quality_summary.md"
MAP_PATH = OUT_DIR / "bus_stop_map.html"
PAGES_INDEX_PATH = BASE_DIR / "index.html"  # published via GitHub Pages
STATS_MAP_PATH = OUT_DIR / "bus_stop_map_stats.html"  # experimental map with a live stats panel
STATS_PAGES_PATH = BASE_DIR / "map-v2.html"  # copy of the above, published via GitHub Pages

# --- Palette (project dataviz conventions) -----------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"
SEQUENTIAL_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
BLUE_CMAP = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUES)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "text.color": TEXT_PRIMARY,
    }
)


# --- 1. Load & clean -----------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


# --- 2 & 3. Summary and data quality report -------------------------------

def build_report(df: pd.DataFrame) -> list[str]:
    lines = ["# Bengaluru Bus Stop Audit - Data Quality Summary", ""]

    lines.append("## Shape")
    lines.append(f"- Rows: {df.shape[0]}")
    lines.append(f"- Columns: {df.shape[1]}")
    lines.append("")

    lines.append("## Column types")
    for dtype, count in df.dtypes.astype(str).value_counts().items():
        lines.append(f"- `{dtype}`: {count} columns")
    lines.append("")

    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    fully_empty = missing[missing == 100].index.tolist()

    lines.append("## Missingness overview")
    lines.append(f"- Columns with 0% missing: {(missing == 0).sum()}")
    lines.append(f"- Columns fully empty (100% missing): {len(fully_empty)}")
    lines.append("")
    lines.append("Top 20 columns by % missing:")
    lines.append("")
    lines.append("| Column | % missing |")
    lines.append("|---|---|")
    for col, pct in missing.head(20).items():
        lines.append(f"| {col} | {pct:.1f}% |")
    lines.append("")

    lines.append("## Data quality issues")
    lines.append("")

    lines.append("### 1. Fully-empty columns (safe to drop)")
    for col in fully_empty:
        lines.append(f"- `{col}`")
    lines.append("")

    lines.append("### 2. Skip-logic missingness (not true gaps)")
    lines.append(
        "These questions are only asked when a bus shelter/signboard exists. "
        "Their ~30-36% missingness lines up with stops recorded as "
        "'No Shelter and No Signboard' or 'Only \"Bus Stop\" Signboard - No Bus "
        "Shelter', not with data collection errors."
    )
    lines.append("")
    conditional_cols = [
        "Condition of Bus Shelter",
        "Does the bus stop have a roof that shelters people from rain?",
        "Does the bus shelter carry advertisements?",
        "Is the seating provided comfortable?",
        "Is the bus stop big enough for the number of people who typically wait here?",
        "If seated at the bus stop, can you clearly see an approaching bus?",
        "How many bus shelters are placed close to each other?",
        "Is the name of the bus stop clearly displayed?",
        "Is bus route information displayed at the stop?",
    ]
    no_shelter_types = {
        "No Shelter and No Signboard",
        'Only "Bus Stop" Signboard - No Bus Shelter',
    }
    lines.append("| Column | % missing | % of missing rows with no shelter/signboard |")
    lines.append("|---|---|---|")
    for col in conditional_cols:
        if col not in df.columns:
            continue
        na_mask = df[col].isna()
        na_pct = na_mask.mean() * 100
        if na_mask.sum() == 0:
            continue
        no_shelter_pct = df.loc[na_mask, "Type of Bus Stop"].isin(no_shelter_types).mean() * 100
        lines.append(f"| {col} | {na_pct:.1f}% | {no_shelter_pct:.1f}% |")
    lines.append("")

    lines.append("### 3. Missing GPS coordinates")
    missing_coords = df[df["latitude"].isna() | df["longitude"].isna()]
    lines.append(
        f"{len(missing_coords)} rows have no usable coordinates "
        f"(`location_flag == 'NO_LOCATION'`) and are excluded from the map "
        f"and any geo-based analysis:"
    )
    lines.append("")
    name_col = (
        "Name of the Bus Stop (as displayed on the shelter, if no info is "
        "available write the commonly known name)"
    )
    for _, row in missing_coords.iterrows():
        lines.append(f"- `{row['_id']}` - {row[name_col]}")
    lines.append("")

    lines.append("### 4. Coordinate reliability")
    src_counts = df["coordinate_source"].value_counts(dropna=False)
    lines.append("Coordinate source breakdown:")
    lines.append("")
    for src, count in src_counts.items():
        lines.append(f"- {src}: {count}")
    lines.append("")
    precision = df["_Survey Location_precision"]
    high_precision_error = df[precision > 100]
    lines.append(
        f"- GPS precision (`_Survey Location_precision`, meters) is only recorded "
        f"for device-collected fixes ({precision.notna().sum()} rows, matching the "
        f"`device_gps` count above). {len(high_precision_error)} of those have a "
        f"reported error greater than 100m, i.e. low-confidence fixes."
    )
    lines.append("")

    lines.append("### 5. Duplicate coordinates with different stop names")
    coord_dupes = df[df.duplicated(subset=["latitude", "longitude"], keep=False)]
    coord_dupes = coord_dupes.dropna(subset=["latitude", "longitude"])
    coord_dupes = coord_dupes.sort_values(["latitude", "longitude"])
    lines.append(
        "Same coordinates recorded under different stop names - likely a "
        "duplicate submission, or a paired directional stop that should be "
        "reviewed manually rather than auto-merged:"
    )
    lines.append("")
    for (lat, lon), group in coord_dupes.groupby(["latitude", "longitude"]):
        names = ", ".join(f"'{n}'" for n in group[name_col])
        lines.append(f"- ({lat:.6f}, {lon:.6f}): {names}")
    lines.append("")

    lines.append("### 6. Duplicate stop names")
    name_norm = df[name_col].str.lower()
    name_dupes = df[name_norm.duplicated(keep=False)].sort_values(name_col)
    lines.append(
        f"{name_dupes[name_col].nunique()} stop names appear more than once "
        f"(case-insensitive exact match), covering {len(name_dupes)} rows. "
        f"Spot checks show some are spelling variants of the same stop "
        f"(e.g. 'Jayanagar' vs 'Jayangar') rather than true repeats - listed "
        f"here for manual review, not auto-merged."
    )
    lines.append("")

    return lines


def write_report(lines: list[str]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote report to {REPORT_PATH}")


# --- 4. Charts -------------------------------------------------------------

def wrap_labels(labels, width=40):
    return [textwrap.fill(str(label), width) for label in labels]


def save_horizontal_bar(counts: pd.Series, colors, filename, title, xlabel="Number of bus stops"):
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(counts) + 1)))
    y = np.arange(len(counts))
    bars = ax.barh(y, counts.values, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(counts.index), color=TEXT_PRIMARY, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, loc="left", pad=14, fontweight="bold")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=TEXT_SECONDARY)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    max_val = max(counts.values)
    is_integer_valued = np.allclose(counts.values, np.round(counts.values))
    for bar, value in zip(bars, counts.values):
        label = f"{value:.0f}" if is_integer_valued else f"{value:.1f}"
        ax.text(
            bar.get_width() + max_val * 0.015,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            color=TEXT_PRIMARY,
            fontsize=9,
        )
    ax.set_xlim(0, max_val * 1.12)
    fig.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def status_bar_chart(df, column, prefix_colors, filename, title, fallback_label=None, fallback_color=MUTED):
    """prefix_colors: ordered list of (label_prefix, color) pairs, matched with
    str.startswith so labels are not tripped up by punctuation variants
    (e.g. dash characters) after the prefix."""
    series = df[column]
    if fallback_label is not None:
        series = series.fillna(fallback_label)
    counts = series.value_counts()

    def rank(label):
        for i, (prefix, _) in enumerate(prefix_colors):
            if label.startswith(prefix):
                return i
        return len(prefix_colors)

    def color_for(label):
        for prefix, color in prefix_colors:
            if label.startswith(prefix):
                return color
        return fallback_color

    counts = counts.reindex(sorted(counts.index, key=rank))
    colors = [color_for(label) for label in counts.index]
    save_horizontal_bar(counts, colors, filename, title)


def chart_type_of_bus_stop(df):
    counts = df["Type of Bus Stop"].value_counts()
    save_horizontal_bar(counts, [BLUE] * len(counts), "01_type_of_bus_stop.png", "Type of Bus Stop")


def chart_shelter_condition(df):
    prefix_colors = [
        ("Clean and Well Maintained", GOOD),
        ("Satisfactory", WARNING),
        ("Unclean/Poorly Maintained", SERIOUS),
        ("Severely Damaged", CRITICAL),
    ]
    status_bar_chart(
        df,
        "Condition of Bus Shelter",
        prefix_colors,
        "02_shelter_condition.png",
        "Condition of Bus Shelter",
        fallback_label="No Shelter (Not Applicable)",
    )


def chart_safety_at_night(df):
    col = "How safe does this bus stop feel, particularly at night?"
    prefix_colors = [
        ("Safe", GOOD),
        ("Somewhat safe", WARNING),
        ("Unsafe", CRITICAL),
        ("Unable to assess", MUTED),
    ]
    status_bar_chart(df, col, prefix_colors, "03_safety_at_night.png", "Perceived Safety at Night")


def chart_encroachment(df):
    status_bar_chart(df, ENCROACHMENT_COL, ENCROACHMENT_PREFIX_COLORS, "04_encroachment.png", "Encroachment Status")


def chart_drainage(df):
    col = "Is there evidence of water-logging or poor drainage around the stop?"
    prefix_colors = [
        ("No", GOOD),
        ("Yes", CRITICAL),
        ("Unsure", MUTED),
    ]
    status_bar_chart(df, col, prefix_colors, "05_drainage.png", "Waterlogging / Drainage")


def chart_land_use(df):
    col = "What is the primary land use or activity near this bus stop?"
    counts = df[col].value_counts()
    save_horizontal_bar(counts, [BLUE] * len(counts), "06_land_use.png", "Primary Land Use Near Bus Stop")


def chart_amenities(df):
    prefix = "Tick all amenities present at this bus stop/"
    cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith("None of the Above")]
    counts = df[cols].sum().sort_values(ascending=False)
    counts.index = [c[len(prefix):] for c in counts.index]
    save_horizontal_bar(counts, [BLUE] * len(counts), "07_amenities.png", "Amenities Present at Bus Stops")


def chart_missingness(df):
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    top = missing.head(15)
    colors = [BLUE_CMAP(v / 100) for v in top.values]
    save_horizontal_bar(top, colors, "08_missingness.png", "Top 15 Columns by % Missing", xlabel="% missing")


def chart_submissions_over_time(df):
    dt = pd.to_datetime(df["Date and Time"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    daily = dt.dt.date.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(daily.index, daily.values, color=BLUE, linewidth=2)
    ax.fill_between(daily.index, daily.values, color=BLUE, alpha=0.15)
    ax.set_title("Bus Stop Audit Submissions Over Time", color=TEXT_PRIMARY, fontsize=13, loc="left", pad=14, fontweight="bold")
    ax.set_ylabel("Stops audited per day", color=TEXT_SECONDARY, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / "09_submissions_over_time.png", dpi=150)
    plt.close(fig)
    print("Saved 09_submissions_over_time.png")


def build_charts(df):
    chart_type_of_bus_stop(df)
    chart_shelter_condition(df)
    chart_safety_at_night(df)
    chart_encroachment(df)
    chart_drainage(df)
    chart_land_use(df)
    chart_amenities(df)
    chart_missingness(df)
    chart_submissions_over_time(df)


# --- 5. Interactive webmap --------------------------------------------------

SAFETY_COL = "How safe does this bus stop feel, particularly at night?"
SAFETY_PREFIX_COLORS = [
    ("Safe", GOOD),
    ("Somewhat safe", WARNING),
    ("Unsafe", CRITICAL),
    ("Unable to assess", MUTED),
]
SAFETY_ORDER = [prefix for prefix, _ in SAFETY_PREFIX_COLORS]

ENCROACHMENT_COL = "Is the bus stop area free of encroachments?"
ENCROACHMENT_PREFIX_COLORS = [
    ("Yes", GOOD),
    ("Partially encroached", WARNING),
    ("Heavily encroached", CRITICAL),
]
ENCROACHMENT_DISPLAY_LABELS = {"Yes": "Free of encroachment"}

LAND_USE_COL = "What is the primary land use or activity near this bus stop?"

SHELTER_CONDITION_COL = "Condition of Bus Shelter"
SHELTER_CONDITION_PREFIX_COLORS = [
    ("Clean and Well Maintained", GOOD),
    ("Satisfactory", WARNING),
    ("Unclean/Poorly Maintained", SERIOUS),
    ("Severely Damaged", CRITICAL),
]
SHELTER_CONDITION_FALLBACK = "No Shelter (Not Applicable)"
SHELTER_CONDITION_ORDER = [p for p, _ in SHELTER_CONDITION_PREFIX_COLORS] + [SHELTER_CONDITION_FALLBACK]

AMENITY_PREFIX = "Tick all amenities present at this bus stop/"
AMENITY_KEYS = {
    "seating": "Seating / bench",
    "sufficient_seating": "Sufficient seating for the number of people typically waiting",
    "dustbin": "Dustbin / waste bin",
    "drinking_water": "Drinking water (tap or dispenser)",
    "toilet_nearby": "Public toilet nearby (within eyesight)",
    "cctv": "CCTV camera",
    "digital_boards": "Digital information boards with real time arrival information and service alerts",
}

DATA_CREDIT_TEXT = "Data Credits: OPenCity Bengaluru Bus Stop Audit - July 2026 (accessed on 26.08.2026)"
GITHUB_URL = "https://github.com/soniadas123"


def safety_color(label):
    for prefix, color in SAFETY_PREFIX_COLORS:
        if isinstance(label, str) and label.startswith(prefix):
            return color
    return MUTED


def shelter_condition_color(label):
    if not isinstance(label, str):
        return MUTED
    for prefix, color in SHELTER_CONDITION_PREFIX_COLORS:
        if label.startswith(prefix):
            return color
    return MUTED


def bucket_for(label, prefix_colors):
    if isinstance(label, str):
        for prefix, _ in prefix_colors:
            if label.startswith(prefix):
                return prefix
    return "Unknown"


LEGEND_HTML = f"""<div id="legend">
  <b>Perceived safety at night</b><br>
  <span style="color:{GOOD};">&#9679;</span> Safe<br>
  <span style="color:{WARNING};">&#9679;</span> Somewhat safe<br>
  <span style="color:{CRITICAL};">&#9679;</span> Unsafe<br>
  <span style="color:{MUTED};">&#9679;</span> Unable to assess
</div>"""

MAP_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; font-family: sans-serif; color: __TEXT_PRIMARY__; }
  body { display: flex; flex-direction: column; height: 100vh; height: 100dvh; }
  header { background: __TEXT_PRIMARY__; color: #fff; padding: 14px 20px; flex: 0 0 auto;
            display: flex; justify-content: space-between; align-items: center; gap: 14px; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header a { color: __LINK_COLOR__; font-size: 12px; text-decoration: none; }
  header a:hover { text-decoration: underline; }
  #map-wrapper { position: relative; width: 100%; flex: 1 1 auto; min-height: 0; }
  #map { position: absolute; inset: 0; }
  #filter-bar { position: absolute; top: 16px; right: 16px; z-index: 1000; width: 200px;
                max-height: calc(100% - 32px); overflow-y: auto;
                background: __SURFACE__; border: 1px solid __GRID__; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.2); padding: 12px 14px;
                display: flex; flex-direction: column; gap: 10px; align-items: stretch; }
  #filter-bar label { font-size: 12px; color: __TEXT_SECONDARY__; display: flex; flex-direction: column; gap: 3px; }
  #filter-bar select { font-size: 13px; padding: 4px 6px; border: 1px solid __GRID__; border-radius: 4px;
                        color: __TEXT_PRIMARY__; background: #fff; width: 100%; }
  #filter-bar button { font-size: 12px; padding: 6px 10px; border: 1px solid __GRID__;
                        border-radius: 4px; background: #fff; color: __TEXT_SECONDARY__; cursor: pointer; }
  #filter-bar button:hover { background: __GRID__; }
  #filter-count { font-size: 12px; color: __TEXT_SECONDARY__; }
  #legend { position: absolute; bottom: 16px; right: 16px; z-index: 1000;
            background: __SURFACE__; padding: 10px 14px; border-radius: 6px;
            border: 1px solid __GRID__; font-size: 13px; color: __TEXT_PRIMARY__;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2); }
  .bus-marker { width: 26px; height: 26px; border-radius: 50%; border: 2px solid #fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.45); display: flex; align-items: center;
                justify-content: center; font-size: 14px; line-height: 1; }
  .leaflet-div-icon { background: transparent; border: none; }
  footer { background: __TEXT_PRIMARY__; color: #d8d7d2; padding: 12px 20px; font-size: 12px;
           flex: 0 0 auto; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px; }
  footer a { color: __LINK_COLOR__; text-decoration: none; }
  footer a:hover { text-decoration: underline; }
</style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <a href="__GITHUB_URL__" target="_blank" rel="noopener">github.com/soniadas123</a>
  </header>
  <div id="map-wrapper">
    <div id="map"></div>
    <div id="filter-bar">
      <label>Corporation
        <select id="filter-corporation"><option value="">All</option></select>
      </label>
      <label>Safety at night
        <select id="filter-safety"><option value="">All</option></select>
      </label>
      <label>Type of bus stop
        <select id="filter-type"><option value="">All</option></select>
      </label>
      <label>Encroachment status
        <select id="filter-encroachment"><option value="">All</option></select>
      </label>
      <label>Land use
        <select id="filter-land-use"><option value="">All</option></select>
      </label>
      <button id="reset-filters" type="button">Reset filters</button>
      <span id="filter-count"></span>
    </div>
    __LEGEND_HTML__
  </div>
  <footer>
    <span>__DATA_CREDIT__</span>
    <span>A Project by Sonia Das &middot; <a href="__GITHUB_URL__" target="_blank" rel="noopener">github.com/soniadas123</a></span>
  </footer>
<script>
const STOPS = __STOPS_JSON__;
const SAFETY_ORDER = __SAFETY_ORDER_JSON__;

const BOUNDARY = __BOUNDARY_GEOJSON__;

const map = L.map('map').setView([__CENTER_LAT__, __CENTER_LON__], 12);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 19
}).addTo(map);

L.geoJSON(BOUNDARY, {
  style: { color: '__TEXT_SECONDARY__', weight: 1.5, dashArray: '4,3', fill: false }
}).addTo(map);

// #map-wrapper's height is set in CSS (viewport-relative), so Leaflet can
// mismeasure it at init time on some browsers. Re-check once layout settles
// and again on resize.
window.addEventListener('load', () => map.invalidateSize());
window.addEventListener('resize', () => map.invalidateSize());
setTimeout(() => map.invalidateSize(), 0);

function busIcon(color) {
  return L.divIcon({
    className: 'bus-marker-wrap',
    html: `<div class="bus-marker" style="background:${color};">&#128652;</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    popupAnchor: [0, -13],
  });
}

function popupHtml(s) {
  return `<div style="font-family: sans-serif; font-size: 13px; max-width: 260px;">
    <b>${s.name}</b><br>
    Type: ${s.type}<br>
    Shelter condition: ${s.condition}
  </div>`;
}

const entries = STOPS.map(s => ({
  stop: s,
  marker: L.marker([s.lat, s.lon], { icon: busIcon(s.color) }).bindPopup(popupHtml(s))
}));

function uniqueSorted(values) {
  return [...new Set(values)].sort();
}

function populateSelect(id, values) {
  const select = document.getElementById(id);
  for (const v of uniqueSorted(values)) {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  }
}

function populateOrdered(id, order, values) {
  const select = document.getElementById(id);
  const present = new Set(values);
  for (const v of order) {
    if (present.has(v)) {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      select.appendChild(opt);
    }
  }
}

populateOrdered('filter-safety', SAFETY_ORDER, STOPS.map(s => s.safety_bucket));
populateSelect('filter-type', STOPS.map(s => s.type));
populateSelect('filter-encroachment', STOPS.map(s => s.encroachment_bucket));
populateSelect('filter-land-use', STOPS.map(s => s.land_use));
populateSelect('filter-corporation', STOPS.map(s => s.corporation));

const filterKeys = {
  'filter-safety': 'safety_bucket',
  'filter-type': 'type',
  'filter-encroachment': 'encroachment_bucket',
  'filter-land-use': 'land_use',
  'filter-corporation': 'corporation',
};
const filterIds = Object.keys(filterKeys);

function applyFilters() {
  const selected = {};
  for (const id of filterIds) {
    selected[id] = document.getElementById(id).value;
  }
  let shown = 0;
  for (const entry of entries) {
    const matches = filterIds.every(id => {
      const value = selected[id];
      return value === "" || entry.stop[filterKeys[id]] === value;
    });
    if (matches) {
      if (!map.hasLayer(entry.marker)) entry.marker.addTo(map);
      shown += 1;
    } else if (map.hasLayer(entry.marker)) {
      map.removeLayer(entry.marker);
    }
  }
  document.getElementById('filter-count').textContent = shown + ' of ' + entries.length + ' stops shown';
}

for (const id of filterIds) {
  document.getElementById(id).addEventListener('change', applyFilters);
}
document.getElementById('reset-filters').addEventListener('click', () => {
  for (const id of filterIds) document.getElementById(id).value = "";
  applyFilters();
});

applyFilters();
</script>
</body>
</html>
"""


def build_map(df):
    name_col = (
        "Name of the Bus Stop (as displayed on the shelter, if no info is "
        "available write the commonly known name)"
    )
    geo_df = df.dropna(subset=["latitude", "longitude"]).copy()
    geometry = [Point(xy) for xy in zip(geo_df["longitude"], geo_df["latitude"])]
    gdf = gpd.GeoDataFrame(geo_df, geometry=geometry, crs="EPSG:4326")

    boundary_gdf = gpd.read_file(BOUNDARY_PATH)[["NewCorp", "geometry"]]
    gdf = gpd.sjoin(gdf, boundary_gdf, how="left", predicate="within").drop(columns=["index_right"])

    stops = []
    for _, row in gdf.iterrows():
        condition = row["Condition of Bus Shelter"]
        condition = str(condition) if isinstance(condition, str) else "No Shelter (Not Applicable)"
        encroachment_bucket = bucket_for(row[ENCROACHMENT_COL], ENCROACHMENT_PREFIX_COLORS)
        stops.append(
            {
                "name": str(row[name_col]),
                "lat": float(row.geometry.y),
                "lon": float(row.geometry.x),
                "type": str(row["Type of Bus Stop"]),
                "land_use": str(row[LAND_USE_COL]),
                "condition": condition,
                "safety_bucket": bucket_for(row[SAFETY_COL], SAFETY_PREFIX_COLORS),
                "color": safety_color(row[SAFETY_COL]),
                "encroachment_bucket": ENCROACHMENT_DISPLAY_LABELS.get(encroachment_bucket, encroachment_bucket),
                "corporation": row["NewCorp"] if isinstance(row["NewCorp"], str) else "Unknown",
            }
        )

    boundary_geojson = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))

    replacements = {
        "__TITLE__": "Bengaluru Bus Stop Audit Map",
        "__TEXT_PRIMARY__": TEXT_PRIMARY,
        "__SURFACE__": SURFACE,
        "__GRID__": GRID,
        "__TEXT_SECONDARY__": TEXT_SECONDARY,
        "__LINK_COLOR__": SEQUENTIAL_BLUES[2],
        "__LEGEND_HTML__": LEGEND_HTML,
        "__DATA_CREDIT__": DATA_CREDIT_TEXT,
        "__GITHUB_URL__": GITHUB_URL,
        "__STOPS_JSON__": json.dumps(stops),
        "__SAFETY_ORDER_JSON__": json.dumps(SAFETY_ORDER),
        "__BOUNDARY_GEOJSON__": json.dumps(boundary_geojson),
        "__CENTER_LAT__": repr(float(gdf.geometry.y.mean())),
        "__CENTER_LON__": repr(float(gdf.geometry.x.mean())),
    }
    html = MAP_HTML_TEMPLATE
    for token, value in replacements.items():
        html = html.replace(token, str(value))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(html, encoding="utf-8")
    PAGES_INDEX_PATH.write_text(html, encoding="utf-8")
    print(f"Saved map to {MAP_PATH} ({len(stops)} stops plotted, {len(df) - len(stops)} excluded for missing coordinates)")
    print(f"Also copied to {PAGES_INDEX_PATH} for GitHub Pages")


# --- 6. Interactive webmap with a live stats panel (experimental, separate output) --------

STATS_MAP_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; font-family: sans-serif; color: __TEXT_PRIMARY__; }
  body { display: flex; flex-direction: column; height: 100vh; height: 100dvh; }
  header { background: __TEXT_PRIMARY__; color: #fff; padding: 14px 20px; flex: 0 0 auto;
            display: flex; justify-content: space-between; align-items: center; gap: 14px; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header a { color: __LINK_COLOR__; font-size: 12px; text-decoration: none; }
  header a:hover { text-decoration: underline; }
  #map-wrapper { position: relative; width: 100%; flex: 1 1 auto; min-height: 0; }
  #map { position: absolute; inset: 0; }
  #filter-bar { position: absolute; top: 16px; right: 16px; z-index: 1000; width: 300px;
                max-width: calc(100% - 32px);
                max-height: calc(100% - 32px); overflow-y: auto;
                background: __SURFACE__; border: 1px solid __GRID__; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.2); padding: 12px 14px;
                display: flex; flex-direction: column; gap: 10px; align-items: stretch; }
  #filter-bar label { font-size: 12px; color: __TEXT_SECONDARY__; display: flex; flex-direction: column; gap: 3px; }
  #filter-bar select { font-size: 13px; padding: 4px 6px; border: 1px solid __GRID__; border-radius: 4px;
                        color: __TEXT_PRIMARY__; background: #fff; width: 100%; }
  #filter-bar button { font-size: 12px; padding: 6px 10px; border: 1px solid __GRID__;
                        border-radius: 4px; background: #fff; color: __TEXT_SECONDARY__; cursor: pointer; }
  #filter-bar button:hover { background: __GRID__; }
  #filter-count { font-size: 12px; color: __TEXT_SECONDARY__; }
  #filter-bar details { display: flex; flex-direction: column; gap: 10px; }
  #filter-bar summary { cursor: pointer; font-weight: 600; font-size: 13px; color: __TEXT_PRIMARY__;
                         margin-bottom: 2px; }
  .filter-group-title { font-size: 12px; color: __TEXT_SECONDARY__; font-weight: 600; margin-top: 2px; }
  .filter-hint { font-size: 11px; color: __TEXT_SECONDARY__; margin: 0 0 2px; line-height: 1.35; }
  .quick-filters { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 2px; }
  .quick-filter-chip { font-size: 11px; padding: 4px 9px; border: 1px solid __GRID__; border-radius: 999px;
                        background: #fff; color: __TEXT_PRIMARY__; cursor: pointer; }
  .quick-filter-chip:hover { background: __GRID__; }
  .quick-filter-chip.active { background: __TEXT_PRIMARY__; color: #fff; border-color: __TEXT_PRIMARY__; }
  .checkbox-row { flex-direction: row !important; align-items: center; gap: 6px; font-size: 12px; }
  #stats-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 4px 0 8px; }
  .stat-card { background: __GRID__; border-radius: 4px; padding: 6px 8px; display: flex; flex-direction: column; }
  .stat-card-value { font-size: 16px; font-weight: 700; color: __TEXT_PRIMARY__; }
  .stat-card-label { font-size: 10px; color: __TEXT_SECONDARY__; }
  .stat-chart-title { font-size: 12.5px; font-weight: 700; color: __TEXT_PRIMARY__; margin: 8px 0 5px; }
  .stat-bar-group { margin-bottom: 4px; }
  .stat-bar-item { margin: 0 0 7px; }
  .stat-bar-label { display: block; font-size: 11px; color: __TEXT_PRIMARY__; line-height: 1.25; margin-bottom: 2px; }
  .stat-bar-row { display: flex; align-items: center; gap: 6px; }
  .stat-bar-track { flex: 1 1 auto; background: __GRID__; height: 14px; overflow: hidden; }
  .stat-bar-fill { height: 100%; }
  .stat-bar-value { flex: 0 0 auto; font-size: 11px; color: __TEXT_SECONDARY__; }
  #legend { position: absolute; bottom: 16px; left: 16px; z-index: 1000;
            background: __SURFACE__; padding: 10px 14px; border-radius: 6px;
            border: 1px solid __GRID__; font-size: 13px; color: __TEXT_PRIMARY__;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2); }
  .bus-marker { width: 26px; height: 26px; border-radius: 50%; border: 2px solid #fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.45); display: flex; align-items: center;
                justify-content: center; font-size: 14px; line-height: 1; }
  .leaflet-div-icon { background: transparent; border: none; }
  footer { background: __TEXT_PRIMARY__; color: #d8d7d2; padding: 12px 20px; font-size: 12px;
           flex: 0 0 auto; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px; }
  footer a { color: __LINK_COLOR__; text-decoration: none; }
  footer a:hover { text-decoration: underline; }
</style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <a href="__GITHUB_URL__" target="_blank" rel="noopener">github.com/soniadas123</a>
  </header>
  <div id="map-wrapper">
    <div id="map"></div>
    <div id="filter-bar">
      <details open>
        <summary>Filters</summary>
        <p class="filter-hint">Not sure where to start? Try a quick filter, or combine the dropdowns and checkboxes below - they all narrow the map together.</p>
        <div class="quick-filters" id="quick-filters"></div>
        <label>Corporation
          <select id="filter-corporation"><option value="">All</option></select>
        </label>
        <label>Safety at night
          <select id="filter-safety"><option value="">All</option></select>
        </label>
        <label>Type of bus stop
          <select id="filter-type"><option value="">All</option></select>
        </label>
        <label>Shelter condition
          <select id="filter-condition"><option value="">All</option></select>
        </label>
        <label>Encroachment status
          <select id="filter-encroachment"><option value="">All</option></select>
        </label>
        <label>Land use
          <select id="filter-land-use"><option value="">All</option></select>
        </label>
        <div class="filter-group">
          <span class="filter-group-title">Amenities (all checked must be present)</span>
          <label class="checkbox-row"><input type="checkbox" data-amenity="seating"> Seating / bench</label>
          <label class="checkbox-row"><input type="checkbox" data-amenity="sufficient_seating"> Sufficient seating</label>
          <label class="checkbox-row"><input type="checkbox" data-amenity="dustbin"> Dustbin</label>
          <label class="checkbox-row"><input type="checkbox" data-amenity="drinking_water"> Drinking water</label>
          <label class="checkbox-row"><input type="checkbox" data-amenity="toilet_nearby"> Public toilet nearby</label>
          <label class="checkbox-row"><input type="checkbox" data-amenity="cctv"> CCTV camera</label>
          <label class="checkbox-row"><input type="checkbox" data-amenity="digital_boards"> Digital info boards</label>
        </div>
        <button id="reset-filters" type="button">Reset filters</button>
        <span id="filter-count"></span>
      </details>
      <details open>
        <summary>Stats (for stops shown)</summary>
        <div id="stats-cards">
          <div class="stat-card"><span class="stat-card-value" id="stat-shown">0</span><span class="stat-card-label">Stops shown</span></div>
          <div class="stat-card"><span class="stat-card-value" id="stat-shelter">0%</span><span class="stat-card-label">Usable shelter</span></div>
          <div class="stat-card"><span class="stat-card-value" id="stat-unsafe">0%</span><span class="stat-card-label">Unsafe at night</span></div>
          <div class="stat-card"><span class="stat-card-value" id="stat-encroach-free">0%</span><span class="stat-card-label">Encroachment-free</span></div>
        </div>
        <div class="stat-bar-group"><span class="stat-chart-title">Shelter condition</span><div id="bars-condition"></div></div>
        <div class="stat-bar-group"><span class="stat-chart-title">Safety at night</span><div id="bars-safety"></div></div>
        <div class="stat-bar-group"><span class="stat-chart-title">Encroachment</span><div id="bars-encroachment"></div></div>
        <div class="stat-bar-group"><span class="stat-chart-title">Amenities coverage</span><div id="bars-amenities"></div></div>
      </details>
    </div>
    __LEGEND_HTML__
  </div>
  <footer>
    <span>__DATA_CREDIT__</span>
    <span>A Project by Sonia Das &middot; <a href="__GITHUB_URL__" target="_blank" rel="noopener">github.com/soniadas123</a></span>
  </footer>
<script>
const STOPS = __STOPS_JSON__;
const SAFETY_ORDER = __SAFETY_ORDER_JSON__;
const SHELTER_ORDER = __SHELTER_ORDER_JSON__;
const SHELTER_FALLBACK = __SHELTER_FALLBACK_JSON__;
const AMENITY_LABELS = __AMENITY_LABELS_JSON__;
const ENCROACHMENT_COLORS = __ENCROACHMENT_COLORS_JSON__;
const BLUE_COLOR = "__BLUE__";

const BOUNDARY = __BOUNDARY_GEOJSON__;

const map = L.map('map').setView([__CENTER_LAT__, __CENTER_LON__], 12);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 19
}).addTo(map);

L.geoJSON(BOUNDARY, {
  style: { color: '__TEXT_SECONDARY__', weight: 1.5, dashArray: '4,3', fill: false }
}).addTo(map);

// #map-wrapper's height is set in CSS (viewport-relative), so Leaflet can
// mismeasure it at init time on some browsers. Re-check once layout settles
// and again on resize.
window.addEventListener('load', () => map.invalidateSize());
window.addEventListener('resize', () => map.invalidateSize());
setTimeout(() => map.invalidateSize(), 0);

function busIcon(color) {
  return L.divIcon({
    className: 'bus-marker-wrap',
    html: `<div class="bus-marker" style="background:${color};">&#128652;</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    popupAnchor: [0, -13],
  });
}

function popupHtml(s) {
  return `<div style="font-family: sans-serif; font-size: 13px; max-width: 260px;">
    <b>${s.name}</b><br>
    Type: ${s.type}<br>
    Shelter condition: ${s.condition}
  </div>`;
}

const entries = STOPS.map(s => ({
  stop: s,
  marker: L.marker([s.lat, s.lon], { icon: busIcon(s.color) }).bindPopup(popupHtml(s))
}));

const conditionColorByLabel = Object.fromEntries(STOPS.map(s => [s.condition, s.condition_color]));
const safetyColorByLabel = Object.fromEntries(STOPS.map(s => [s.safety_bucket, s.color]));

function uniqueSorted(values) {
  return [...new Set(values)].sort();
}

function populateSelect(id, values) {
  const select = document.getElementById(id);
  for (const v of uniqueSorted(values)) {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  }
}

function populateOrdered(id, order, values) {
  const select = document.getElementById(id);
  const present = new Set(values);
  for (const v of order) {
    if (present.has(v)) {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      select.appendChild(opt);
    }
  }
}

populateOrdered('filter-safety', SAFETY_ORDER, STOPS.map(s => s.safety_bucket));
populateSelect('filter-type', STOPS.map(s => s.type));
populateOrdered('filter-condition', SHELTER_ORDER, STOPS.map(s => s.condition));
populateSelect('filter-encroachment', STOPS.map(s => s.encroachment_bucket));
populateSelect('filter-land-use', STOPS.map(s => s.land_use));
populateSelect('filter-corporation', STOPS.map(s => s.corporation));

const filterKeys = {
  'filter-safety': 'safety_bucket',
  'filter-type': 'type',
  'filter-condition': 'condition',
  'filter-encroachment': 'encroachment_bucket',
  'filter-land-use': 'land_use',
  'filter-corporation': 'corporation',
};
const filterIds = Object.keys(filterKeys);

function checkedAmenities() {
  return [...document.querySelectorAll('#filter-bar input[type=checkbox][data-amenity]:checked')]
    .map(cb => cb.dataset.amenity);
}

function resetFilterControls() {
  for (const id of filterIds) document.getElementById(id).value = "";
  for (const cb of document.querySelectorAll('#filter-bar input[type=checkbox][data-amenity]')) cb.checked = false;
}

function setSelectByPrefix(id, prefix) {
  const select = document.getElementById(id);
  for (const opt of select.options) {
    if (opt.value.startsWith(prefix)) { select.value = opt.value; return; }
  }
}

const QUICK_FILTERS = [
  { label: 'Unsafe at night', apply: () => setSelectByPrefix('filter-safety', 'Unsafe') },
  { label: 'No shelter at all', apply: () => { document.getElementById('filter-condition').value = SHELTER_FALLBACK; } },
  { label: 'Severely damaged shelters', apply: () => setSelectByPrefix('filter-condition', 'Severely Damaged') },
  { label: 'Heavily encroached', apply: () => setSelectByPrefix('filter-encroachment', 'Heavily encroached') },
  { label: 'Has CCTV', apply: () => { document.querySelector('[data-amenity="cctv"]').checked = true; } },
];

const quickFilterBar = document.getElementById('quick-filters');
for (const [i, qf] of QUICK_FILTERS.entries()) {
  const chip = document.createElement('button');
  chip.type = 'button';
  chip.className = 'quick-filter-chip';
  chip.textContent = qf.label;
  chip.dataset.index = i;
  chip.addEventListener('click', () => {
    const alreadyActive = chip.classList.contains('active');
    resetFilterControls();
    for (const el of quickFilterBar.children) el.classList.remove('active');
    if (!alreadyActive) {
      qf.apply();
      chip.classList.add('active');
    }
    applyFilters();
  });
  quickFilterBar.appendChild(chip);
}

function clearActiveQuickFilter() {
  for (const el of quickFilterBar.children) el.classList.remove('active');
}

function countBy(items, keyFn) {
  const counts = new Map();
  for (const item of items) {
    const k = keyFn(item);
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  return counts;
}

function renderBarGroup(containerId, rows) {
  const container = document.getElementById(containerId);
  const max = Math.max(1, ...rows.map(r => r.value));
  container.innerHTML = rows.map(r => `
    <div class="stat-bar-item">
      <span class="stat-bar-label">${r.label}</span>
      <div class="stat-bar-row">
        <div class="stat-bar-track">
          <div class="stat-bar-fill" style="width:${(r.value / max * 100).toFixed(1)}%; background:${r.color};"></div>
        </div>
        <span class="stat-bar-value">${r.value}</span>
      </div>
    </div>`).join('');
}

function renderStats(shown) {
  const total = shown.length;
  const pct = n => total ? Math.round(n / total * 100) : 0;

  document.getElementById('stat-shown').textContent = total;
  document.getElementById('stat-shelter').textContent = pct(shown.filter(s => s.condition !== SHELTER_FALLBACK).length) + '%';
  document.getElementById('stat-unsafe').textContent = pct(shown.filter(s => s.safety_bucket === 'Unsafe').length) + '%';
  document.getElementById('stat-encroach-free').textContent = pct(shown.filter(s => s.encroachment_bucket === 'Free of encroachment').length) + '%';

  const conditionCounts = countBy(shown, s => s.condition);
  renderBarGroup('bars-condition', SHELTER_ORDER
    .filter(k => conditionCounts.has(k))
    .map(k => ({ label: k, value: conditionCounts.get(k), color: conditionColorByLabel[k] })));

  const safetyCounts = countBy(shown, s => s.safety_bucket);
  renderBarGroup('bars-safety', SAFETY_ORDER
    .filter(k => safetyCounts.has(k))
    .map(k => ({ label: k, value: safetyCounts.get(k), color: safetyColorByLabel[k] })));

  const encroachCounts = countBy(shown, s => s.encroachment_bucket);
  renderBarGroup('bars-encroachment', [...encroachCounts.entries()]
    .map(([label, value]) => ({ label, value, color: ENCROACHMENT_COLORS[label] || BLUE_COLOR })));

  const amenityRows = Object.entries(AMENITY_LABELS)
    .map(([key, label]) => ({ label, value: shown.filter(s => s.amenities[key]).length, color: BLUE_COLOR }))
    .sort((a, b) => b.value - a.value);
  renderBarGroup('bars-amenities', amenityRows);
}

function applyFilters() {
  const selected = {};
  for (const id of filterIds) {
    selected[id] = document.getElementById(id).value;
  }
  const requiredAmenities = checkedAmenities();
  let shown = 0;
  const shownStops = [];
  for (const entry of entries) {
    const matchesDropdowns = filterIds.every(id => {
      const value = selected[id];
      return value === "" || entry.stop[filterKeys[id]] === value;
    });
    const matchesAmenities = requiredAmenities.every(key => entry.stop.amenities[key]);
    const matches = matchesDropdowns && matchesAmenities;
    if (matches) {
      if (!map.hasLayer(entry.marker)) entry.marker.addTo(map);
      shown += 1;
      shownStops.push(entry.stop);
    } else if (map.hasLayer(entry.marker)) {
      map.removeLayer(entry.marker);
    }
  }
  document.getElementById('filter-count').textContent = shown + ' of ' + entries.length + ' stops shown';
  renderStats(shownStops);
}

for (const id of filterIds) {
  document.getElementById(id).addEventListener('change', () => { clearActiveQuickFilter(); applyFilters(); });
}
for (const cb of document.querySelectorAll('#filter-bar input[type=checkbox][data-amenity]')) {
  cb.addEventListener('change', () => { clearActiveQuickFilter(); applyFilters(); });
}
document.getElementById('reset-filters').addEventListener('click', () => {
  resetFilterControls();
  clearActiveQuickFilter();
  applyFilters();
});

applyFilters();
</script>
</body>
</html>
"""


def build_stats_map(df):
    """Same map as build_map(), plus a live stats panel and two extra filters
    (shelter condition, amenities). Writes to a separate file - does not
    touch bus_stop_map.html or index.html."""
    name_col = (
        "Name of the Bus Stop (as displayed on the shelter, if no info is "
        "available write the commonly known name)"
    )
    geo_df = df.dropna(subset=["latitude", "longitude"]).copy()
    geometry = [Point(xy) for xy in zip(geo_df["longitude"], geo_df["latitude"])]
    gdf = gpd.GeoDataFrame(geo_df, geometry=geometry, crs="EPSG:4326")

    boundary_gdf = gpd.read_file(BOUNDARY_PATH)[["NewCorp", "geometry"]]
    gdf = gpd.sjoin(gdf, boundary_gdf, how="left", predicate="within").drop(columns=["index_right"])

    stops = []
    for _, row in gdf.iterrows():
        condition = row[SHELTER_CONDITION_COL]
        condition = str(condition) if isinstance(condition, str) else SHELTER_CONDITION_FALLBACK
        encroachment_bucket = bucket_for(row[ENCROACHMENT_COL], ENCROACHMENT_PREFIX_COLORS)
        amenities = {key: bool(row[AMENITY_PREFIX + col]) for key, col in AMENITY_KEYS.items()}
        stops.append(
            {
                "name": str(row[name_col]),
                "lat": float(row.geometry.y),
                "lon": float(row.geometry.x),
                "type": str(row["Type of Bus Stop"]),
                "land_use": str(row[LAND_USE_COL]),
                "condition": condition,
                "condition_color": shelter_condition_color(condition),
                "safety_bucket": bucket_for(row[SAFETY_COL], SAFETY_PREFIX_COLORS),
                "color": safety_color(row[SAFETY_COL]),
                "encroachment_bucket": ENCROACHMENT_DISPLAY_LABELS.get(encroachment_bucket, encroachment_bucket),
                "corporation": row["NewCorp"] if isinstance(row["NewCorp"], str) else "Unknown",
                "amenities": amenities,
            }
        )

    boundary_geojson = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))

    encroachment_colors = {
        ENCROACHMENT_DISPLAY_LABELS.get(prefix, prefix): color
        for prefix, color in ENCROACHMENT_PREFIX_COLORS
    }

    replacements = {
        "__TITLE__": "Bengaluru Bus Stop Audit Map - Live Stats",
        "__TEXT_PRIMARY__": TEXT_PRIMARY,
        "__SURFACE__": SURFACE,
        "__GRID__": GRID,
        "__TEXT_SECONDARY__": TEXT_SECONDARY,
        "__LINK_COLOR__": SEQUENTIAL_BLUES[2],
        "__LEGEND_HTML__": LEGEND_HTML,
        "__DATA_CREDIT__": DATA_CREDIT_TEXT,
        "__GITHUB_URL__": GITHUB_URL,
        "__STOPS_JSON__": json.dumps(stops),
        "__SAFETY_ORDER_JSON__": json.dumps(SAFETY_ORDER),
        "__SHELTER_ORDER_JSON__": json.dumps(SHELTER_CONDITION_ORDER),
        "__SHELTER_FALLBACK_JSON__": json.dumps(SHELTER_CONDITION_FALLBACK),
        "__AMENITY_LABELS_JSON__": json.dumps(AMENITY_KEYS),
        "__ENCROACHMENT_COLORS_JSON__": json.dumps(encroachment_colors),
        "__BLUE__": BLUE,
        "__BOUNDARY_GEOJSON__": json.dumps(boundary_geojson),
        "__CENTER_LAT__": repr(float(gdf.geometry.y.mean())),
        "__CENTER_LON__": repr(float(gdf.geometry.x.mean())),
    }
    html = STATS_MAP_HTML_TEMPLATE
    for token, value in replacements.items():
        html = html.replace(token, str(value))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_MAP_PATH.write_text(html, encoding="utf-8")
    STATS_PAGES_PATH.write_text(html, encoding="utf-8")
    print(f"Saved stats map to {STATS_MAP_PATH} ({len(stops)} stops plotted, {len(df) - len(stops)} excluded for missing coordinates)")
    print(f"Also copied to {STATS_PAGES_PATH} for GitHub Pages")


# --- Main --------------------------------------------------------------

def main():
    df = load_data()
    report_lines = build_report(df)
    write_report(report_lines)
    build_charts(df)
    build_map(df)
    build_stats_map(df)


if __name__ == "__main__":
    main()
