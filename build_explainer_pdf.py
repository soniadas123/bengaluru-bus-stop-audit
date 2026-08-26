"""
Builds the plain-language explainer PDF for the Bengaluru bus stop audit
dataset, embedding a few of the charts produced by eda_bus_stop_audit.py.

Body text renders in Times New Roman, 12pt, with 6mm spacing both between
wrapped lines within a paragraph and between paragraphs/blocks.

Run with:
    python build_explainer_pdf.py
"""

from html.parser import HTMLParser
from pathlib import Path

from fpdf import FPDF

BASE_DIR = Path(__file__).resolve().parent
CONTENT_PATH = BASE_DIR / "explainer_content.html"
CHART_DIR = BASE_DIR / "outputs" / "charts"
OUT_PATH = BASE_DIR / "outputs" / "Understanding_the_Bengaluru_Bus_Stop_Audit_Data.pdf"

FONT_FAMILY = "Times"
BODY_SIZE = 12
SPACING = 6  # mm: line height within a block, and the gap left after it

BLOCK_TAGS = {"h1", "h2", "h3", "p", "li"}

CHART_FIGURES = [
    ("01_type_of_bus_stop.png", "Figure 1. Type of bus stop across all 406 audited stops."),
    ("03_safety_at_night.png", "Figure 2. Perceived safety at night, the field used to color the webmap."),
    ("08_missingness.png", "Figure 3. The 15 columns with the most missing data (see the caveats above for why)."),
]


class ContentParser(HTMLParser):
    """Pulls out block-level text and the chart-placement marker. Inline
    tags (b, i) are dropped since each block renders in a single style."""

    def __init__(self):
        super().__init__()
        self.blocks = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        if tag in BLOCK_TAGS:
            self._current = {"tag": tag, "text": ""}

    def handle_endtag(self, tag):
        if tag in BLOCK_TAGS and self._current is not None:
            text = " ".join(self._current["text"].split())
            if text:
                self.blocks.append({"tag": tag, "text": text})
            self._current = None

    def handle_data(self, data):
        if self._current is not None:
            self._current["text"] += data

    def handle_comment(self, data):
        if data.strip() == "CHARTS_HERE":
            self.blocks.append({"tag": "charts_marker"})


def parse_blocks():
    parser = ContentParser()
    parser.feed(CONTENT_PATH.read_text(encoding="utf-8"))
    return parser.blocks


def add_chart_pages(pdf: FPDF):
    for filename, caption in CHART_FIGURES:
        pdf.add_page()
        pdf.image(str(CHART_DIR / filename), x=20, w=170)
        pdf.ln(4)
        pdf.set_font(FONT_FAMILY, style="I", size=10)
        pdf.multi_cell(0, SPACING, caption)


HEADING_SIZES = {"h1": 20, "h2": 15, "h3": BODY_SIZE}


def render_block(pdf: FPDF, block):
    tag = block["tag"]
    text = block["text"]
    if tag in HEADING_SIZES:
        pdf.set_font(FONT_FAMILY, style="B", size=HEADING_SIZES[tag])
        pdf.multi_cell(0, SPACING, text)
    elif tag == "li":
        pdf.set_font(FONT_FAMILY, size=BODY_SIZE)
        pdf.multi_cell(0, SPACING, f"-  {text}")
    else:
        pdf.set_font(FONT_FAMILY, size=BODY_SIZE)
        pdf.multi_cell(0, SPACING, text)
    pdf.ln(SPACING)


def build_pdf():
    blocks = parse_blocks()

    pdf = FPDF(format="A4")
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=BODY_SIZE)

    for block in blocks:
        if block["tag"] == "charts_marker":
            add_chart_pages(pdf)
            pdf.add_page()
            pdf.set_font(FONT_FAMILY, size=BODY_SIZE)
        else:
            render_block(pdf, block)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT_PATH))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    build_pdf()
