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
import numpy as np
import time
from collections import Counter
from typing import List, Tuple
import tempfile

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

GRAPH_STYLES = ["Bar Chart", "Horizontal Bar", "Pie Chart", "Dot Plot", "Stacked Bar"]

# ── Model loading ─────────────────────────────────────────────────────────────
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

def save_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv",
                                      mode="w", encoding="utf-8", newline="")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name

# ── Chart: distribution (Chart 1) ─────────────────────────────────────────────

def make_distribution_chart(label_counter: Counter, library: str, style: str) -> plt.Figure:
    names  = [l for l in WNUT_LABELS if l in label_counter]
    counts = [label_counter[l] for l in names]
    colors = [LABEL_COLORS[l] for l in names]

    if style == "Pie Chart":
        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(
            counts, labels=names, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=1.5)
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title(f"{library} — Entity Type Distribution", fontweight="bold", pad=15)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(7, 4))

    if style == "Horizontal Bar":
        bars = ax.barh(names, counts, color=colors)
        ax.set_xlabel("Count")
        ax.set_facecolor("#f0f4f8")
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", fontsize=9)

    elif style == "Dot Plot":
        ax.hlines(names, 0, counts, colors=colors, linewidth=2, linestyle="--", alpha=0.5)
        ax.scatter(counts, names, color=colors, s=120, zorder=5)
        ax.set_xlabel("Count")
        ax.set_facecolor("#fafafa")
        ax.xaxis.grid(True, linestyle=":", alpha=0.6)
        for name, cnt in zip(names, counts):
            ax.text(cnt + 0.3, name, str(cnt), va="center", fontsize=9)

    elif style == "Stacked Bar":
        left = 0
        for name, cnt, col in zip(names, counts, colors):
            ax.barh(["Entities"], [cnt], left=left, color=col, label=name,
                    edgecolor="white", linewidth=1.2)
            if cnt > 0:
                ax.text(left + cnt / 2, 0, f"{name}\n{cnt}",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            left += cnt
        ax.set_xlabel("Count")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_facecolor("#f5f5f5")
        ax.yaxis.set_visible(False)

    else:  # Bar Chart
        bars = ax.bar(names, counts, color=colors)
        ax.set_ylabel("Count")
        ax.set_facecolor("#f9f9f9")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, str(cnt),
                    ha="center", va="bottom", fontsize=9)

    ax.set_title(f"{library} — Entity Type Distribution", fontweight="bold")
    fig.tight_layout()
    return fig

# ── Chart: top-N per entity (Chart 2) ─────────────────────────────────────────

def make_topn_chart(counter: Counter, entity_filter: List[str],
                    top_n: int, library: str, style: str) -> plt.Figure:
    n = len(entity_filter)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4.5 * n), squeeze=False)

    for i, label in enumerate(entity_filter):
        ax = axes[i][0]
        entries = {t: c for (t, l), c in counter.items() if l == label}
        color   = LABEL_COLORS.get(label, "#333333")

        if not entries:
            ax.text(0.5, 0.5, f"No '{label}' entities found.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.axis("off")
            continue

        top = sorted(entries.items(), key=lambda x: -x[1])[:top_n]
        xlabels, counts = zip(*top)

        if style == "Pie Chart":
            wedge_colors = [plt.cm.tab20(j / len(xlabels)) for j in range(len(xlabels))]
            ax.pie(counts, labels=xlabels, colors=wedge_colors,
                   autopct="%1.1f%%", startangle=140,
                   wedgeprops=dict(edgecolor="white", linewidth=1.2))
            ax.set_title(f"Top {top_n} '{label}' — {library}", fontweight="bold")

        elif style == "Horizontal Bar":
            bars = ax.barh(xlabels, counts, color=color)
            ax.set_xlabel("Count")
            ax.set_facecolor("#f0f4f8")
            ax.xaxis.grid(True, linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)
            for bar, cnt in zip(bars, counts):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        str(cnt), va="center", fontsize=8)
            ax.set_title(f"Top {top_n} '{label}' — {library}", fontweight="bold")

        elif style == "Dot Plot":
            y_pos = range(len(xlabels))
            ax.hlines(y_pos, 0, counts, colors=color, linewidth=2, linestyle="--", alpha=0.5)
            ax.scatter(counts, y_pos, color=color, s=100, zorder=5)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(xlabels, fontsize=8)
            ax.set_xlabel("Count")
            ax.set_facecolor("#fafafa")
            ax.xaxis.grid(True, linestyle=":", alpha=0.6)
            for y, cnt in zip(y_pos, counts):
                ax.text(cnt + 0.1, y, str(cnt), va="center", fontsize=8)
            ax.set_title(f"Top {top_n} '{label}' — {library}", fontweight="bold")

        elif style == "Stacked Bar":
            left = 0
            bar_colors = [plt.cm.tab20(j / max(len(xlabels), 1)) for j in range(len(xlabels))]
            for lbl, cnt, col in zip(xlabels, counts, bar_colors):
                ax.barh([""], [cnt], left=left, color=col,
                        edgecolor="white", linewidth=1.0)
                if cnt > 0:
                    ax.text(left + cnt / 2, 0, f"{lbl}\n{cnt}",
                            ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")
                left += cnt
            ax.set_xlabel("Count")
            ax.yaxis.set_visible(False)
            ax.set_facecolor("#f5f5f5")
            ax.set_title(f"Top {top_n} '{label}' — {library}", fontweight="bold")

        else:  # Bar Chart
            bars = ax.bar(range(len(xlabels)), counts, color=color)
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=9)
            ax.set_ylabel("Count")
            ax.set_facecolor("#f9f9f9")
            ax.yaxis.grid(True, linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)
            for bar, cnt in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1, str(cnt),
                        ha="center", va="bottom", fontsize=8)
            ax.set_title(f"Top {top_n} '{label}' — {library}", fontweight="bold")

    fig.tight_layout(pad=3.0)
    return fig

# ── Comparison charts ─────────────────────────────────────────────────────────

def make_comparison_dist_chart(lc_spacy: Counter, lc_stanza: Counter, style: str) -> plt.Figure:
    names  = WNUT_LABELS
    s_vals = [lc_spacy.get(l, 0)  for l in names]
    t_vals = [lc_stanza.get(l, 0) for l in names]

    if style == "Pie Chart":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = [LABEL_COLORS[l] for l in names]
        for ax, vals, lib in [(ax1, s_vals, "spaCy"), (ax2, t_vals, "Stanza")]:
            non_zero = [(n, v, c) for n, v, c in zip(names, vals, colors) if v > 0]
            if non_zero:
                ns, vs, cs = zip(*non_zero)
                ax.pie(vs, labels=ns, colors=cs, autopct="%1.1f%%", startangle=140,
                       wedgeprops=dict(edgecolor="white", linewidth=1.5))
            ax.set_title(f"{lib} — Distribution", fontweight="bold")
        fig.suptitle("spaCy vs Stanza — Entity Type Distribution", fontweight="bold")
        fig.tight_layout()
        return fig

    if style == "Stacked Bar":
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = [LABEL_COLORS[l] for l in names]
        for lib_vals, lib_name in [(s_vals, "spaCy"), (t_vals, "Stanza")]:
            left = 0
            for name, cnt, col in zip(names, lib_vals, colors):
                ax.barh([lib_name], [cnt], left=left, color=col,
                        edgecolor="white", linewidth=1.2)
                if cnt > 0:
                    ax.text(left + cnt / 2, lib_name,
                            f"{name}\n{cnt}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
                left += cnt
        ax.set_xlabel("Count")
        ax.set_title("spaCy vs Stanza — Entity Type Distribution", fontweight="bold")
        ax.set_facecolor("#f5f5f5")
        fig.tight_layout()
        return fig

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))

    if style == "Horizontal Bar":
        b1 = ax.barh(x - w/2, s_vals, w, label="spaCy",  color="#4C72B0")
        b2 = ax.barh(x + w/2, t_vals, w, label="Stanza", color="#DD8452")
        ax.set_yticks(x)
        ax.set_yticklabels(names)
        ax.set_xlabel("Count")
        ax.set_facecolor("#f0f4f8")
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        for bar in list(b1) + list(b2):
            h = bar.get_width()
            if h:
                ax.text(h + 0.2, bar.get_y() + bar.get_height() / 2,
                        str(int(h)), va="center", fontsize=8)

    elif style == "Dot Plot":
        ax.hlines(x - 0.1, 0, s_vals, colors="#4C72B0", linewidth=2, linestyle="--", alpha=0.5)
        ax.hlines(x + 0.1, 0, t_vals, colors="#DD8452", linewidth=2, linestyle="--", alpha=0.5)
        ax.scatter(s_vals, x - 0.1, color="#4C72B0", s=100, zorder=5, label="spaCy")
        ax.scatter(t_vals, x + 0.1, color="#DD8452", s=100, zorder=5, label="Stanza")
        ax.set_yticks(x)
        ax.set_yticklabels(names)
        ax.set_xlabel("Count")
        ax.set_facecolor("#fafafa")
        ax.xaxis.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    else:  # Bar Chart
        b1 = ax.bar(x - w/2, s_vals, w, label="spaCy",  color="#4C72B0")
        b2 = ax.bar(x + w/2, t_vals, w, label="Stanza", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Count")
        ax.set_facecolor("#f9f9f9")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                        str(int(h)), ha="center", va="bottom", fontsize=8)
        ax.legend()

    ax.set_title("spaCy vs Stanza — Entity Type Distribution", fontweight="bold")
    fig.tight_layout()
    return fig


def make_comparison_topn_chart(c_spacy: Counter, c_stanza: Counter,
                                entity_filter: List[str], top_n: int, style: str) -> plt.Figure:
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

        combined = sorted(all_keys, key=lambda k: -(s_entries.get(k,0)+t_entries.get(k,0)))[:top_n]
        sv = [s_entries.get(k, 0) for k in combined]
        tv = [t_entries.get(k, 0) for k in combined]
        x  = np.arange(len(combined))
        w  = 0.35

        if style == "Pie Chart":
            pie_colors = [plt.cm.tab20(j / max(len(combined),1)) for j in range(len(combined))]
            non_zero_s = [(k, v, c) for k, v, c in zip(combined, sv, pie_colors) if v > 0]
            non_zero_t = [(k, v, c) for k, v, c in zip(combined, tv, pie_colors) if v > 0]
            if non_zero_s or non_zero_t:
                ax.axis("off")
                ax.set_title(f"Top {top_n} '{label}' — spaCy vs Stanza (Pie)", fontweight="bold")
                if non_zero_s:
                    ks, vs, cs = zip(*non_zero_s)
                    inset1 = fig.add_axes([0.05, 0.1 + (n-1-i)/n * 0.9,  0.4, 0.8/n])
                    inset1.pie(vs, labels=ks, colors=cs, autopct="%1.0f%%",
                               wedgeprops=dict(edgecolor="white"))
                    inset1.set_title("spaCy", fontsize=8)
                if non_zero_t:
                    ks, vs, cs = zip(*non_zero_t)
                    inset2 = fig.add_axes([0.55, 0.1 + (n-1-i)/n * 0.9, 0.4, 0.8/n])
                    inset2.pie(vs, labels=ks, colors=cs, autopct="%1.0f%%",
                               wedgeprops=dict(edgecolor="white"))
                    inset2.set_title("Stanza", fontsize=8)
            continue

        elif style == "Horizontal Bar":
            b1 = ax.barh(x - w/2, sv, w, label="spaCy",  color="#4C72B0")
            b2 = ax.barh(x + w/2, tv, w, label="Stanza", color="#DD8452")
            ax.set_yticks(x)
            ax.set_yticklabels(combined, fontsize=8)
            ax.set_xlabel("Count")
            ax.set_facecolor("#f0f4f8")
            ax.xaxis.grid(True, linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)
            for bar in list(b1) + list(b2):
                h = bar.get_width()
                if h:
                    ax.text(h + 0.1, bar.get_y() + bar.get_height() / 2,
                            str(int(h)), va="center", fontsize=7)

        elif style == "Dot Plot":
            ax.hlines(x - 0.1, 0, sv, colors="#4C72B0", linewidth=2, linestyle="--", alpha=0.5)
            ax.hlines(x + 0.1, 0, tv, colors="#DD8452", linewidth=2, linestyle="--", alpha=0.5)
            ax.scatter(sv, x - 0.1, color="#4C72B0", s=90, zorder=5, label="spaCy")
            ax.scatter(tv, x + 0.1, color="#DD8452", s=90, zorder=5, label="Stanza")
            ax.set_yticks(x)
            ax.set_yticklabels(combined, fontsize=8)
            ax.set_xlabel("Count")
            ax.set_facecolor("#fafafa")
            ax.xaxis.grid(True, linestyle=":", alpha=0.6)

        elif style == "Stacked Bar":
            s_left = t_left = 0
            bar_colors = [plt.cm.tab20(j / max(len(combined),1)) for j in range(len(combined))]
            for k, s, t, col in zip(combined, sv, tv, bar_colors):
                ax.barh(["spaCy"],  [s], left=s_left, color=col, edgecolor="white")
                ax.barh(["Stanza"], [t], left=t_left, color=col, edgecolor="white")
                if s > 0:
                    ax.text(s_left + s/2, "spaCy",  f"{k}\n{s}",
                            ha="center", va="center", fontsize=6, color="white")
                if t > 0:
                    ax.text(t_left + t/2, "Stanza", f"{k}\n{t}",
                            ha="center", va="center", fontsize=6, color="white")
                s_left += s
                t_left += t
            ax.set_xlabel("Count")
            ax.set_facecolor("#f5f5f5")

        else:  # Bar Chart
            b1 = ax.bar(x - w/2, sv, w, label="spaCy",  color="#4C72B0")
            b2 = ax.bar(x + w/2, tv, w, label="Stanza", color="#DD8452")
            ax.set_xticks(x)
            ax.set_xticklabels(combined, rotation=35, ha="right", fontsize=9)
            ax.set_ylabel("Count")
            ax.set_facecolor("#f9f9f9")
            ax.yaxis.grid(True, linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)
            for bar in list(b1) + list(b2):
                h = bar.get_height()
                if h:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                            str(int(h)), ha="center", va="bottom", fontsize=8)

        ax.set_title(f"Top {top_n} '{label}' — spaCy vs Stanza", fontweight="bold")
        ax.legend()

    fig.tight_layout(pad=3.0)
    return fig

# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(file_objs, library: str, entity_filter: List[str],
            top_n: int, show_table: bool, graph_style: str):

    if not file_objs:
        return None, None, None, gr.update(visible=False), "⚠️ Please upload at least one .txt file."
    if not entity_filter:
        return None, None, None, gr.update(visible=False), "⚠️ Please select at least one entity type."

    text = read_files(file_objs)
    if not text.strip():
        return None, None, None, gr.update(visible=False), "⚠️ Uploaded file(s) are empty."

    top_n      = int(top_n)
    comparison = (library == "Both (Comparison)")

    # ── Single library ────────────────────────────────────────────────────────
    if not comparison:
        t0  = time.time()
        raw = run_spacy(text) if library == "spaCy" else run_stanza(text)
        elapsed = time.time() - t0

        counter: Counter       = Counter()
        label_counter: Counter = Counter()
        for surface, label, _ in raw:
            counter[(surface.strip(), label)] += 1
            label_counter[label] += 1

        status = (f"✅ **{library}** — {sum(label_counter.values())} entities found "
                  f"across {len(label_counter)} type(s) in {elapsed:.1f}s.")

        fig1 = make_distribution_chart(label_counter, library, graph_style)
        fig2 = make_topn_chart(counter, entity_filter, top_n, library, graph_style)

        rows = [{"Entity": t, "Type": l, "Count": c}
                for (t, l), c in counter.items() if l in entity_filter]
        rows.sort(key=lambda x: (WNUT_LABELS.index(x["Type"]), -x["Count"]))
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Entity","Type","Count"])

        csv_path  = save_csv(df)
        table_out = gr.update(value=df, visible=show_table)
        dl_out    = gr.update(value=csv_path, visible=show_table)

        return fig1, fig2, table_out, dl_out, status

    # ── Comparison ────────────────────────────────────────────────────────────
    t0    = time.time()
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

    fig1 = make_comparison_dist_chart(lc_sp, lc_st, graph_style)
    fig2 = make_comparison_topn_chart(c_sp, c_st, entity_filter, top_n, graph_style)

    all_keys = set(c_sp) | set(c_st)
    rows = []
    for (t, l) in all_keys:
        if l in entity_filter:
            rows.append({
                "Entity":       t,
                "Type":         l,
                "spaCy Count":  c_sp.get((t, l), 0),
                "Stanza Count": c_st.get((t, l), 0),
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
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📄 Upload .txt file(s) — drag multiple files at once",
                file_types=[".txt"],
                type="filepath",
                file_count="multiple",
            )
            library_radio = gr.Radio(
                choices=["spaCy", "Stanza", "Both (Comparison)"],
                value="spaCy",
                label="NER Library",
            )
            graph_style = gr.Radio(
                choices=GRAPH_STYLES,
                value="Bar Chart",
                label="Graph Style",
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

        with gr.Column(scale=2):
            status_box    = gr.Markdown("Upload file(s) and click **Run Analysis**.")
            chart_dist    = gr.Plot(label="Entity Type Distribution")
            chart_topn    = gr.Plot(label="Top-N Entity Mentions")
            results_table = gr.Dataframe(label="Results Table", visible=False, wrap=True)
            download_btn  = gr.File(label="⬇ Download results as CSV", visible=False)

    select_all_btn.click(fn=select_all_entities, inputs=[], outputs=[entity_select])

    run_btn.click(
        fn=analyze,
        inputs=[file_input, library_radio, entity_select,
                top_n_slider, show_table_chk, graph_style],
        outputs=[chart_dist, chart_topn, results_table, download_btn, status_box],
    )

demo.launch()