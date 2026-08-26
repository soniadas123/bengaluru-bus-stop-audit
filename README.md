# Bengaluru Bus Stop Audit

An exploratory analysis of a volunteer-collected accessibility and safety audit of 406 bus
stops across Bengaluru, surveyed between June 7 and July 17, 2026 using KoboToolbox. The
data collection was a group effort by volunteers who visited each stop in person; the analysis,
charts, interactive map, and this write-up are my own work.

**Live map:** https://soniadas123.github.io/bengaluru-bus-stop-audit/

## What's in this repo

- `bengaluru_bus_stop_audit_2026.csv` - the raw survey export (406 rows, 86 columns).
- `blr_boundary.geojson` / `blr_boundary.qmd` - Bengaluru's five BBMP corporation zone
  boundaries, used to tag each stop with its corporation on the map.
- `eda_bus_stop_audit.py` - reads the CSV and produces everything in `outputs/`, plus
  `index.html` (a copy of the map, published via GitHub Pages).
- `explainer_content.html` / `build_explainer_pdf.py` - source and build script for the PDF
  explainer.
- `outputs/`
  - `data_quality_summary.md` - a written summary of the dataset's shape, missing data, and
    the data quality issues found during review (duplicate stops, missing coordinates,
    skip-logic gaps, and so on).
  - `charts/` - nine PNG charts covering shelter type, condition, safety, encroachment,
    drainage, land use, amenities, missingness, and audit progress over time.
  - `bus_stop_map.html` - the interactive Leaflet map (same file as `index.html`).
  - `Understanding_the_Bengaluru_Bus_Stop_Audit_Data.pdf` - a plain-language explainer of the
    dataset's fields, its caveats, and how to use the map.

## Running it yourself

The script needs `pandas`, `geopandas`, `shapely`, and `matplotlib` (the map itself is
hand-built with Leaflet, loaded from a CDN in the generated HTML), plus `fpdf2` for the PDF:

```
python eda_bus_stop_audit.py       # data quality report, charts, and the map
python build_explainer_pdf.py      # the PDF explainer
```

## Data credits

Data Credits: OpenCity Bengaluru Bus Stop Audit - July 2026 (accessed on 26.08.2026)

## Contributors

- **Sonia Das** - analysis, charts, interactive map, and this write-up -
  [github.com/soniadas123](https://github.com/soniadas123)
