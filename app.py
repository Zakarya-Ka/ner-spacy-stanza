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
import pandas as pd
import time, math
from collections import Counter
from typing import List, Tuple, Dict
import io, os

# ── Label taxonomy (matches the paper) ──────────────────────────────────────
WNUT_LABELS = ["person", "location", "corporation", "product", "creative-work", "group"]

SPACY2WNUT = {
    "PERSON":      "person",
    "ORG":         "corporation",
    "NORP":        "group",
    "GPE":         "location",
    "LOC":         "location",
    "PRODUCT":     "product",
    "WORK_OF_ART": "creative-work",
}

STANZA2WNUT = {
    "PERSON":      "person",
    "ORG":         "corporation",
    "GPE":         "location",
    "LOC":         "location",
    "NORP":        "group",
    "PRODUCT":     "product",
    "WORK_OF_ART": "creative-work",
}

# ── Model loading (cached so it only loads once) ─────────────────────────────
_spacy_nlp  = None
_stanza_nlp = None

def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], check=True)
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

def get_stanza():
    global _stanza_nlp
    if _stanza_nlp is None:
        stanza.download("en", processors="tokenize,ner", verbose=False)
        _stanza_nlp = stanza.Pipeline(
            "en",
            processors="tokenize,ner",
            use_gpu=False,
            verbose=False,
        )
    return _stanza_nlp

# ── Core NER functions ────────────────────────────────────────────────────────

def run_spacy(text: str) -> List[Tuple[str, str, str]]:
    """Returns list of (surface_text, mapped_label, original_label)."""
    nlp  = get_spacy()
    doc  = nlp(text)
    results = []
    for ent in doc.ents:
        mapped = SPACY2WNUT.get(ent.label_)
        if mapped:
            results.append((ent.text, mapped, ent.label_))
    return results

def run_stanza(text: str) -> List[Tuple[str, str, str]]:
    nlp = get_stanza()
    doc = nlp(text)
    results = []
    for sent in doc.sentences:
        for ent in sent.ents:
            mapped = STANZA2WNUT.get(ent.type)
            if mapped:
                results.append((ent.text, mapped, ent.type))
    return results

# ── Chart builder ─────────────────────────────────────────────────────────────

def make_bar_chart(
    counter: Counter,
    title: str,
    top_n: int,
    entity_filter: List[str],
    library: str,
) -> plt.Figure:
    """Bar chart of top-N surface forms, optionally filtered by entity type(s)."""
    filtered = {
        text: count
        for (text, label), count in counter.items()
        if label in entity_filter
    }
    if not filtered:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No entities found for the selected type(s).",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
        ax.axis("off")
        return fig

    top = sorted(filtered.items(), key=lambda x: -x[1])[:top_n]
    labels_x, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(max(8, min(top_n, 25) * 0.55 + 2), 5))
    bars = ax.bar(range(len(labels_x)), counts,
                  color="#4C72B0" if library == "spaCy" else "#DD8452")
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=12, fontweight="bold")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, str(count),
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig

# ── Main analysis function ────────────────────────────────────────────────────

def analyze(
    file_obj,
    library: str,
    entity_filter: List[str],
    top_n: int,
    show_table: bool,
):
    # ── Input validation ──────────────────────────────────────────────────────
    if file_obj is None:
        return (
            None, None, None,
            gr.update(visible=False),
            "⚠️ Please upload a .txt file.",
        )

    if not entity_filter:
        return (
            None, None, None,
            gr.update(visible=False),
            "⚠️ Please select at least one entity type.",
        )

    # Read file
    if hasattr(file_obj, "name"):
        with open(file_obj.name, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = file_obj.decode("utf-8", errors="replace")

    if not text.strip():
        return None, None, None, gr.update(visible=False), "⚠️ The uploaded file is empty."

    top_n = int(top_n)

    # ── Run NER ──────────────────────────────────────────────────────────────
    t0 = time.time()
    if library == "spaCy":
        raw = run_spacy(text)
    else:
        raw = run_stanza(text)
    elapsed = time.time() - t0

    if not raw:
        return (
            None, None, None, gr.update(visible=False),
            f"✅ Done in {elapsed:.1f}s — no mapped entities found in the text."
        )

    # ── Build counters ────────────────────────────────────────────────────────
    counter: Counter = Counter()
    label_counter: Counter = Counter()
    for surface, label, _ in raw:
        counter[(surface.strip(), label)] += 1
        label_counter[label] += 1

    total_entities = sum(label_counter.values())
    status_msg = (
        f"✅ **{library}** finished in **{elapsed:.1f}s** — "
        f"found **{total_entities}** mapped entities across "
        f"{len(label_counter)} label(s)."
    )

    # ── Chart 1: label distribution (all labels, always shown) ───────────────
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    lab_names  = list(label_counter.keys())
    lab_counts = list(label_counter.values())
    color = "#4C72B0" if library == "spaCy" else "#DD8452"
    bars1 = ax1.bar(lab_names, lab_counts, color=color)
    ax1.set_title(f"{library} — Entity Type Distribution", fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Entity Type")
    for bar, cnt in zip(bars1, lab_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3, str(cnt),
                 ha="center", va="bottom", fontsize=9)
    fig1.tight_layout()

    # ── Chart 2: one subplot per selected entity type ─────────────────────────
    # Palette — one distinct colour per WNUT label
    LABEL_COLORS = {
        "person":        "#4C72B0",
        "location":      "#DD8452",
        "corporation":   "#55A868",
        "product":       "#C44E52",
        "creative-work": "#8172B2",
        "group":         "#937860",
    }

    n_selected = len(entity_filter)
    fig2, axes = plt.subplots(
        n_selected, 1,
        figsize=(10, 4.5 * n_selected),
        squeeze=False,
    )

    for row_idx, label in enumerate(entity_filter):
        ax = axes[row_idx][0]

        # Get top-N surface forms for this label only
        label_entries = {
            text: cnt
            for (text, lbl), cnt in counter.items()
            if lbl == label
        }

        if not label_entries:
            ax.text(0.5, 0.5, f"No '{label}' entities found.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Top {top_n} — {label}", fontweight="bold")
            ax.axis("off")
            continue

        top_items = sorted(label_entries.items(), key=lambda x: -x[1])[:top_n]
        x_labels, counts = zip(*top_items)
        bar_color = LABEL_COLORS.get(label, "#333333")

        bars = ax.bar(range(len(x_labels)), counts, color=bar_color)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title(f"Top {top_n} '{label}' mentions — {library}", fontweight="bold")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1, str(cnt),
                    ha="center", va="bottom", fontsize=8)

    fig2.tight_layout(pad=3.0)

    # ── Table (all entities matching selected filters) ────────────────────────
    table_data = [
        {"Entity": text, "Type": label, "Count": cnt}
        for (text, label), cnt in counter.items()
        if label in entity_filter
    ]
    table_data.sort(key=lambda x: (-x["Count"], x["Type"]))
    df = pd.DataFrame(table_data) if table_data else pd.DataFrame(
        columns=["Entity", "Type", "Count"]
    )

    table_out = gr.update(value=df, visible=show_table)

    return fig1, fig2, df if show_table else None, table_out, status_msg


# ── Gradio UI ─────────────────────────────────────────────────────────────────


with gr.Blocks(title="NER Analysis — spaCy & Stanza", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔍 NER Analysis — spaCy vs Stanza
    Upload a plain-text `.txt` file and explore Named Entity Recognition results
    using **spaCy** (`en_core_web_trf`) or **Stanza** (`en` NER pipeline).

    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📄 Upload .txt file",
                file_types=[".txt"],
                type="filepath",
                height=100,
            )
            library_radio = gr.Radio(
                choices=["spaCy", "Stanza"],
                value="spaCy",
                label="NER Library",
            )
            entity_select = gr.CheckboxGroup(
                choices=WNUT_LABELS,
                value=["person"],
                label="Select entity types to analyse (pick one or more)",
            )
            top_n_slider = gr.Slider(
                minimum=1, maximum=25, step=1, value=10,
                label="Top N entities to show in bar chart (max 25)",
            )
            show_table_chk = gr.Checkbox(
                value=False,
                label="Also show full results table",
            )
            run_btn = gr.Button("▶ Run Analysis", variant="primary")

        with gr.Column(scale=2):
            status_box  = gr.Markdown("Upload a file and click **Run Analysis**.")
            chart_dist  = gr.Plot(label="Entity Type Distribution")
            chart_top_n = gr.Plot(label="Top-N Entity Mentions")
            results_table = gr.Dataframe(
                label="Full Results Table",
                visible=False,
                wrap=True,
            )

    run_btn.click(
        fn=analyze,
        inputs=[
            file_input,
            library_radio,
            entity_select,
            top_n_slider,
            show_table_chk,
        ],
        outputs=[
            chart_dist,
            chart_top_n,
            results_table,
            results_table,
            status_box,
        ],
    )

    gr.Markdown("""
    ---
    **Entity label mapping** (from library-native labels → WNUT'17 taxonomy):

    | Library label | WNUT label |
    |---|---|
    | PERSON | person |
    | ORG | corporation |
    | GPE / LOC | location |
    | NORP | group |
    | PRODUCT | product |
    | WORK_OF_ART | creative-work |
    """)

demo.launch()