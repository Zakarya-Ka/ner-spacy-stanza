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
import time
from collections import Counter
from typing import List, Tuple
import tempfile, os

# ── Label taxonomy ────────────────────────────────────────────────────────────
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

LABEL_COLORS = {
    "person":        "#4C72B0",
    "location":      "#DD8452",
    "corporation":   "#55A868",
    "product":       "#C44E52",
    "creative-work": "#8172B2",
    "group":         "#937860",
}

# ── Model loading (cached) ────────────────────────────────────────────────────
_spacy_nlp  = None
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
def run_spacy(text: str) -> List[Tuple[str, str, str]]:
    nlp = get_spacy()
    doc = nlp(text)
    return [(ent.text, SPACY2WNUT[ent.label_], ent.label_)
            for ent in doc.ents if ent.label_ in SPACY2WNUT]

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

def read_files(file_objs) -> str:
    """Read one or more uploaded files and concatenate their text."""
    if file_objs is None:
        return ""
    if not isinstance(file_objs, list):
        file_objs = [file_objs]
    texts = []
    for f in file_objs:
        path = f.name if hasattr(f, "name") else f
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            texts.append(fh.read())
    return "\n\n".join(texts)

# ── Chart builders ────────────────────────────────────────────────────────────
def make_distribution_chart(label_counter: Counter, library: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = list(label_counter.keys())
    counts = list(label_counter.values())
    color  = "#4C72B0" if library == "spaCy" else "#DD8452"
    bars   = ax.bar(names, counts, color=color)
    ax.set_title(f"{library} — Entity Type Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(cnt),
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig

def make_topn_chart(counter: Counter, entity_filter: List[str],
                    top_n: int, library: str) -> plt.Figure:
    n = len(entity_filter)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4.5 * n), squeeze=False)
    for i, label in enumerate(entity_filter):
        ax = axes[i][0]
        entries = {t: c for (t, l), c in counter.items() if l == label}
        if not entries:
            ax.text(0.5, 0.5, f"No '{label}' entities found.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Top {top_n} — {label}", fontweight="bold")
            ax.axis("off")
            continue
        top = sorted(entries.items(), key=lambda x: -x[1])[:top_n]
        xlabels, counts = zip(*top)
        bars = ax.bar(range(len(xlabels)), counts, color=LABEL_COLORS.get(label, "#333"))
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title(f"Top {top_n} '{label}' — {library}", fontweight="bold")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1, str(cnt),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout(pad=3.0)
    return fig

def make_comparison_dist_chart(lc_spacy: Counter, lc_stanza: Counter) -> plt.Figure:
    """Side-by-side distribution bars for both libraries."""
    import numpy as np
    labels = WNUT_LABELS
    s_vals = [lc_spacy.get(l, 0)  for l in labels]
    t_vals = [lc_stanza.get(l, 0) for l in labels]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w/2, s_vals, w, label="spaCy",  color="#4C72B0")
    b2 = ax.bar(x + w/2, t_vals, w, label="Stanza", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("spaCy vs Stanza — Entity Type Distribution", fontweight="bold")
    ax.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig

def make_comparison_topn_chart(c_spacy: Counter, c_stanza: Counter,
                                entity_filter: List[str], top_n: int) -> plt.Figure:
    """For each selected entity: side-by-side bars spaCy vs Stanza."""
    import numpy as np
    n = len(entity_filter)
    fig, axes = plt.subplots(n, 1, figsize=(11, 5 * n), squeeze=False)
    for i, label in enumerate(entity_filter):
        ax = axes[i][0]
        s_entries = {t: c for (t, l), c in c_spacy.items()  if l == label}
        t_entries = {t: c for (t, l), c in c_stanza.items() if l == label}
        all_keys  = set(s_entries) | set(t_entries)
        if not all_keys:
            ax.text(0.5, 0.5, f"No '{label}' entities found in either library.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.axis("off")
            continue
        # rank by combined count, take top_n
        combined = sorted(all_keys, key=lambda k: -(s_entries.get(k,0)+t_entries.get(k,0)))[:top_n]
        x  = np.arange(len(combined))
        w  = 0.35
        sv = [s_entries.get(k, 0) for k in combined]
        tv = [t_entries.get(k, 0) for k in combined]
        b1 = ax.bar(x - w/2, sv, w, label="spaCy",  color="#4C72B0")
        b2 = ax.bar(x + w/2, tv, w, label="Stanza", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(combined, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title(f"Top {top_n} '{label}' — spaCy vs Stanza", fontweight="bold")
        ax.legend()
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, str(int(h)),
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout(pad=3.0)
    return fig

# ── CSV download helper ───────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w",
                                      encoding="utf-8", newline="")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name

# ── Main analysis function ────────────────────────────────────────────────────
def analyze(file_objs, library: str, entity_filter: List[str], top_n: int, show_table: bool):

    # Validation
    if not file_objs:
        return None, None, None, gr.update(visible=False), gr.update(visible=False), "⚠️ Please upload at least one .txt file."
    if not entity_filter:
        return None, None, None, gr.update(visible=False), gr.update(visible=False), "⚠️ Please select at least one entity type."

    text = read_files(file_objs)
    if not text.strip():
        return None, None, None, gr.update(visible=False), gr.update(visible=False), "⚠️ Uploaded file(s) are empty."

    top_n = int(top_n)
    comparison = (library == "Both (Comparison)")

    # ── Single library mode ───────────────────────────────────────────────────
    if not comparison:
        t0  = time.time()
        raw = run_spacy(text) if library == "spaCy" else run_stanza(text)
        elapsed = time.time() - t0

        counter: Counter = Counter()
        label_counter: Counter = Counter()
        for surface, label, _ in raw:
            counter[(surface.strip(), label)] += 1
            label_counter[label] += 1

        status = (f"✅ **{library}** — {sum(label_counter.values())} entities found "
                  f"across {len(label_counter)} type(s) in {elapsed:.1f}s.")

        fig1 = make_distribution_chart(label_counter, library)
        fig2 = make_topn_chart(counter, entity_filter, top_n, library)

        # Table — sorted by type (WNUT order) then count descending
        rows = [{"Entity": t, "Type": l, "Count": c}
                for (t, l), c in counter.items() if l in entity_filter]
        rows.sort(key=lambda x: (WNUT_LABELS.index(x["Type"]), -x["Count"]))
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Entity","Type","Count"])

        csv_path = save_csv(df)
        table_out  = gr.update(value=df,       visible=show_table)
        dl_out     = gr.update(value=csv_path, visible=show_table)

        return fig1, fig2, table_out, dl_out, status

    # ── Comparison mode ───────────────────────────────────────────────────────
    t0 = time.time()
    raw_s = run_spacy(text)
    raw_t = run_stanza(text)
    elapsed = time.time() - t0

    def build_counters(raw):
        counter: Counter = Counter()
        lc: Counter      = Counter()
        for surface, label, _ in raw:
            counter[(surface.strip(), label)] += 1
            lc[label] += 1
        return counter, lc

    c_sp, lc_sp = build_counters(raw_s)
    c_st, lc_st = build_counters(raw_t)

    status = (f"✅ **Comparison** done in {elapsed:.1f}s — "
              f"spaCy: {sum(lc_sp.values())} entities | "
              f"Stanza: {sum(lc_st.values())} entities.")

    fig1 = make_comparison_dist_chart(lc_sp, lc_st)
    fig2 = make_comparison_topn_chart(c_sp, c_st, entity_filter, top_n)

    # Combined table
    all_keys = set(c_sp) | set(c_st)
    rows = []
    for (t, l) in all_keys:
        if l in entity_filter:
            rows.append({
                "Entity":        t,
                "Type":          l,
                "spaCy Count":   c_sp.get((t, l), 0),
                "Stanza Count":  c_st.get((t, l), 0),
            })
    rows.sort(key=lambda x: (WNUT_LABELS.index(x["Type"]),
                              -(x["spaCy Count"] + x["Stanza Count"])))
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Entity","Type","spaCy Count","Stanza Count"])

    csv_path  = save_csv(df)
    table_out = gr.update(value=df,       visible=show_table)
    dl_out    = gr.update(value=csv_path, visible=show_table)

    return fig1, fig2, table_out, dl_out, status

# ── UI ────────────────────────────────────────────────────────────────────────
def select_all_entities():
    return WNUT_LABELS

with gr.Blocks(title="NER Analysis — spaCy & Stanza", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔍 NER Analysis — spaCy & Stanza
    Upload one or more `.txt` files and explore Named Entity Recognition results.
    > Based on *Kaouch (2025) — A Comparative Analysis of Off-the-Shelf NER Libraries on Noisy Text*.
    """)

    with gr.Row():
        # ── Left panel ────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📄 Upload .txt file(s) — drag multiple files at once",
                file_types=[".txt"],
                type="filepath",
                file_count="multiple",
                height=100,
            )
            library_radio = gr.Radio(
                choices=["spaCy", "Stanza", "Both (Comparison)"],
                value="spaCy",
                label="NER Library",
            )

            with gr.Row():
                entity_select = gr.CheckboxGroup(
                    choices=WNUT_LABELS,
                    value=["person"],
                    label="Entity types",
                )
                select_all_btn = gr.Button("Select All", size="sm", scale=0)

            top_n_slider = gr.Slider(
                minimum=1, maximum=25, step=1, value=10,
                label="Top N entities in chart (max 25)",
            )
            show_table_chk = gr.Checkbox(value=False, label="Show & download results table")
            run_btn = gr.Button("▶ Run Analysis", variant="primary")

        # ── Right panel ───────────────────────────────────────────────────────
        with gr.Column(scale=2):
            status_box    = gr.Markdown("Upload file(s) and click **Run Analysis**.")
            chart_dist    = gr.Plot(label="Entity Type Distribution")
            chart_topn    = gr.Plot(label="Top-N Entity Mentions")
            results_table = gr.Dataframe(label="Results Table", visible=False, wrap=True)
            download_btn  = gr.File(label="⬇ Download results as CSV", visible=False)

    # ── Events ────────────────────────────────────────────────────────────────
    select_all_btn.click(fn=select_all_entities, inputs=[], outputs=[entity_select])

    run_btn.click(
        fn=analyze,
        inputs=[file_input, library_radio, entity_select, top_n_slider, show_table_chk],
        outputs=[chart_dist, chart_topn, results_table, download_btn, status_box],
    )

    gr.Markdown("""
    ---
    **Entity label mapping** (library-native → WNUT'17 taxonomy):

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