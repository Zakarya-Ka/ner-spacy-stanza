"""
NER Analysis Space — spaCy & Stanza on user-uploaded .txt files
Based on methodology from: Kaouch (2025), "A Comparative Analysis of
Off-the-Shelf NER Libraries on Noisy Text"
"""

import gradio as gr
import spacy
import stanza
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import time
from collections import Counter
from typing import List, Tuple
import tempfile

# ── Label taxonomy ────────────────────────────────────────────────────────────
WNUT_LABELS = ["person", "location", "corporation", "product", "creative-work", "group"]

SPACY2WNUT = {
    "PERSON": "person", "ORG": "corporation", "NORP": "group",
    "GPE": "location", "LOC": "location", "PRODUCT": "product",
    "WORK_OF_ART": "creative-work",
}
STANZA2WNUT = {
    "PERSON": "person", "ORG": "corporation", "GPE": "location",
    "LOC": "location", "NORP": "group", "PRODUCT": "product",
    "WORK_OF_ART": "creative-work",
}

LABEL_COLORS = {
    "person": "#4C72B0", "location": "#DD8452", "corporation": "#55A868",
    "product": "#C44E52", "creative-work": "#8172B2", "group": "#937860",
}

GRAPH_STYLES = [
    "Bar Chart",
    "Horizontal Bar Chart",
    "Pie Chart",
    "Dot Plot",
    "Treemap",
]

# ── Model loading ─────────────────────────────────────────────────────────────
_spacy_nlp = None
_stanza_nlp = None

def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

def get_stanza():
    global _stanza_nlp
    if _stanza_nlp is None:
        stanza.download("en", processors="tokenize,ner", verbose=False)
        _stanza_nlp = stanza.Pipeline("en", processors="tokenize,ner", use_gpu=False, verbose=False)
    return _stanza_nlp

# ── Core NER ──────────────────────────────────────────────────────────────────
def run_spacy(text):
    nlp = get_spacy()
    doc = nlp(text)
    return [(ent.text, SPACY2WNUT[ent.label_], ent.label_)
            for ent in doc.ents if ent.label_ in SPACY2WNUT]

def run_stanza(text):
    nlp = get_stanza()
    doc = nlp(text)
    results = []
    for sent in doc.sentences:
        for ent in sent.ents:
            mapped = STANZA2WNUT.get(ent.type)
            if mapped:
                results.append((ent.text, mapped, ent.type))
    return results

def read_files(file_objs) -> str:
    if not file_objs:
        return ""
    if not isinstance(file_objs, list):
        file_objs = [file_objs]
    texts = []
    for f in file_objs:
        path = f.name if hasattr(f, "name") else f
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            texts.append(fh.read())
    return "\n\n".join(texts)

def build_counters(raw):
    counter, lc = Counter(), Counter()
    for surface, label, _ in raw:
        counter[(surface.strip(), label)] += 1
        lc[label] += 1
    return counter, lc

def save_csv(df):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv",
                                      mode="w", encoding="utf-8", newline="")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name

# ── Individual graph renderers ────────────────────────────────────────────────

def draw_bar(ax, xlabels, counts, color, title):
    bars = ax.bar(range(len(xlabels)), counts, color=color)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, str(cnt),
                ha="center", va="bottom", fontsize=8)

def draw_hbar(ax, xlabels, counts, color, title):
    y = range(len(xlabels))
    bars = ax.barh(list(y), counts, color=color)
    ax.set_yticks(list(y))
    ax.set_yticklabels(xlabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title, fontweight="bold")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=8)

def draw_pie(ax, xlabels, counts, colors, title):
    wedge_colors = [colors[i % len(colors)] for i in range(len(xlabels))]
    wedges, texts, autotexts = ax.pie(
        counts, labels=xlabels, autopct="%1.1f%%",
        colors=wedge_colors, startangle=140,
        textprops={"fontsize": 8},
    )
    ax.set_title(title, fontweight="bold")

def draw_dot(ax, xlabels, counts, color, title):
    y = range(len(xlabels))
    ax.hlines(list(y), 0, counts, color="#cccccc", linewidth=1)
    ax.plot(counts, list(y), "o", color=color, markersize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(xlabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title, fontweight="bold")
    for cnt, yi in zip(counts, y):
        ax.text(cnt + 0.15, yi, str(cnt), va="center", fontsize=8)

def draw_treemap(ax, xlabels, counts, colors, title):
    """Simple squarified-style treemap using nested rectangles."""
    total = sum(counts)
    if total == 0:
        ax.axis("off")
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontweight="bold")
    # Simple row-based layout
    sorted_items = sorted(zip(counts, xlabels, colors), reverse=True)
    x, y_pos, row_h = 0.0, 1.0, 0.0
    row_items, row_total = [], 0
    cell_width = 1.0

    def draw_row(items, y_top, height):
        rx = 0.0
        for cnt, lbl, col in items:
            w = cnt / total
            rect = mpatches.FancyBboxPatch(
                (rx + 0.005, y_top - height + 0.005),
                w - 0.01, height - 0.01,
                boxstyle="round,pad=0.01",
                facecolor=col, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)
            if w > 0.06 and height > 0.06:
                ax.text(rx + w / 2, y_top - height / 2,
                        f"{lbl}\n{cnt}", ha="center", va="center",
                        fontsize=max(6, min(9, int(w * 60))),
                        color="white", fontweight="bold")
            rx += w

    # Stack rows of roughly equal area
    row_items, row_sum, y_cur = [], 0, 1.0
    for cnt, lbl, col in sorted_items:
        row_items.append((cnt, lbl, col))
        row_sum += cnt
        row_h = row_sum / total
        if row_h >= 0.25 or (cnt, lbl, col) == sorted_items[-1]:
            draw_row(row_items, y_cur, row_h)
            y_cur -= row_h
            row_items, row_sum = [], 0

def render_single(counter, entity_filter, top_n, library, style):
    """One figure with one subplot per entity, in the chosen style."""
    n = len(entity_filter)
    fig_h = 4.5 if style in ("Bar Chart", "Horizontal Bar Chart", "Dot Plot") else 5
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h * n), squeeze=False)

    for i, label in enumerate(entity_filter):
        ax = axes[i][0]
        entries = {t: c for (t, l), c in counter.items() if l == label}
        color   = LABEL_COLORS.get(label, "#333")
        title   = f"Top {top_n} '{label}' — {library}"

        if not entries:
            ax.text(0.5, 0.5, f"No '{label}' entities found.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.axis("off")
            continue

        top      = sorted(entries.items(), key=lambda x: -x[1])[:top_n]
        xlabels  = [x[0] for x in top]
        counts   = [x[1] for x in top]
        pie_cols = [LABEL_COLORS.get(l, "#888") for l in WNUT_LABELS]

        if style == "Bar Chart":
            draw_bar(ax, xlabels, counts, color, title)
        elif style == "Horizontal Bar Chart":