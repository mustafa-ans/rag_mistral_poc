#!/usr/bin/env python3
"""
score_eval.py — cross-validate a stress-test run against the gold reference,
print the metrics, and render a bar chart.

Reads three files from an evaluation folder (default: evaluation/stress_75):
  - gold.json             : expected FAQ targets + behavior per question (answer key)
  - retrieval_log.jsonl   : what the pipeline retrieved (written by main.py eval / retrieval-only mode)
  - evaluation_results.csv: your manual labels (optional; for answer-quality metrics)

Computes automatically from the log:
  - retrieval recall@k for in-scope questions, overall and per category
  - multi-part "both retrieved" vs "either retrieved"
  - out-of-scope and adversarial refusal rates

Usage:
  python evaluation/score_eval.py
  python evaluation/score_eval.py --dir evaluation/stress_75
  python evaluation/score_eval.py --no-chart           # skip the PNG
  python evaluation/score_eval.py --chart-path out.png

Matching is case-insensitive substring: an expected selector counts as retrieved if it
appears inside any retrieved FAQ question text.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

CATEGORY_ORDER = ["paraphrase", "jargon", "hard_negative", "multi_part",
                  "typo", "rambling", "ambiguous", "negation"]


def load_gold(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, {item["idx"]: item for item in data["items"]}


def load_log(path):
    by_idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_idx[rec["idx"]] = rec
    return by_idx


def expected_matches(selectors, retrieved_faqs):
    rl = [q.lower() for q in retrieved_faqs]
    return [s for s in selectors if any(s.lower() in q for q in rl)]


def compute(gold, log):
    """Return a structured results dict from gold + retrieval log."""
    cat_total = defaultdict(int)
    cat_hit = defaultdict(int)
    r = {
        "inscope_hit": 0, "inscope_total": 0,
        "multi_both": 0, "multi_either": 0, "multi_total": 0,
        "oos_refused": 0, "oos_total": 0,
        "adv_refused": 0, "adv_total": 0,
        "missing": [],
        "categories": {},
    }
    for idx, item in sorted(gold.items()):
        entry = log.get(idx)
        if entry is None:
            r["missing"].append(idx)
            continue
        retrieved = [x["faq"] for x in entry.get("retrieved", [])]
        refused = entry.get("refused", False)
        behavior = item["behavior"]
        cat = item["category"]

        if behavior in ("refuse", "deflect"):
            if behavior == "refuse":
                r["oos_total"] += 1
                r["oos_refused"] += int(refused)
            else:
                r["adv_total"] += 1
                r["adv_refused"] += int(refused)
            continue

        selectors = item["expected"]
        match = item.get("match", "any")
        found = expected_matches(selectors, retrieved)
        hit = (len(found) == len(selectors) and selectors) if match == "all" else (len(found) >= 1)

        r["inscope_total"] += 1
        r["inscope_hit"] += int(bool(hit))
        cat_total[cat] += 1
        cat_hit[cat] += int(bool(hit))
        if cat == "multi_part":
            r["multi_total"] += 1
            r["multi_both"] += int(len(found) == len(selectors))
            r["multi_either"] += int(len(found) >= 1)

    for cat in CATEGORY_ORDER:
        if cat_total[cat]:
            r["categories"][cat] = (cat_hit[cat], cat_total[cat])
    return r


def load_human_labels(csv_path):
    cols = {"retrieval_success": [], "answer_correct": [], "no_hallucination_on_oos": []}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for c in cols:
                v = (row.get(c) or "").strip()
                if v in ("0", "1"):
                    cols[c].append(int(v))
    return {c: (sum(v), len(v)) for c, v in cols.items() if v}


def pct(hit_total):
    hit, total = hit_total
    return f"{(100 * hit / total):.0f}% ({hit}/{total})" if total else "n/a (0)"


def print_report(r, human, k, min_sim, folder):
    print("=" * 64)
    print(f" Stress-test scoring   (k={k}, min_sim={min_sim})")
    print(f" folder: {folder}")
    print("=" * 64)
    if r["missing"]:
        m = r["missing"]
        print(f"\n[!] {len(m)} gold questions had no log entry (idx {m[:10]}"
              f"{'...' if len(m) > 10 else ''}). Re-run over the full set.")

    print("\n## Retrieval (automatic, log vs gold.json)")
    print(f"  In-scope recall@{k}: {pct((r['inscope_hit'], r['inscope_total']))}")
    print("\n  By category:")
    for cat, ht in r["categories"].items():
        print(f"    {cat:<14} {pct(ht)}")
    if r["multi_total"]:
        print(f"\n  Multi-part — both FAQs: {pct((r['multi_both'], r['multi_total']))}; "
              f"at least one: {pct((r['multi_either'], r['multi_total']))}")

    print("\n## Safety / scope (automatic = did it decline to retrieve?)")
    print(f"  Out-of-scope refusal: {pct((r['oos_refused'], r['oos_total']))}")
    print(f"  Adversarial refusal:  {pct((r['adv_refused'], r['adv_total']))}")
    print("  Note: this measures retrieval suppression only. Whether the *answer* safely")
    print("  declined a question that still retrieved context is the human-label call")
    print("  (no_hallucination_on_oos in the CSV).")

    if human:
        print("\n## Human labels (evaluation_results.csv)")
        for c, ht in human.items():
            print(f"  {c:<24} {pct(ht)}")
    else:
        print("\n## Human labels: none found (run labeling mode to add them).")
    print("\n" + "=" * 64)


def make_chart(r, human, k, out_path):
    """Render per-category recall + refusal rates (+ human labels) to a PNG."""
    labels, values, colors = [], [], []

    def add(label, ht, color):
        hit, total = ht
        if total:
            labels.append(label)
            values.append(100 * hit / total)
            colors.append(color)

    add(f"Overall\nrecall@{k}", (r["inscope_hit"], r["inscope_total"]), "#2563eb")
    for cat, ht in r["categories"].items():
        # highlight hard_negative — the category most likely to be weak
        add(cat, ht, "#dc2626" if cat == "hard_negative" else "#60a5fa")
    add("OOS\nrefusal", (r["oos_refused"], r["oos_total"]), "#16a34a")
    add("Adversarial\nrefusal", (r["adv_refused"], r["adv_total"]), "#16a34a")
    if human:
        for c, ht in human.items():
            short = {"retrieval_success": "label:\nretrieval",
                     "answer_correct": "label:\nanswer",
                     "no_hallucination_on_oos": "label:\nno-halluc"}.get(c, c)
            add(short, ht, "#a855f7")

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("percent")
    ax.set_title(f"RAG stress-test scorecard (k={k})  —  red = hard negatives")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="evaluation/stress_75",
                    help="folder with gold.json, retrieval_log.jsonl, evaluation_results.csv")
    ap.add_argument("--no-chart", dest="chart", action="store_false",
                    help="skip the PNG chart")
    ap.add_argument("--chart-path", default=None, help="output path for the chart PNG")
    ap.set_defaults(chart=True)
    args = ap.parse_args()

    gold_path = os.path.join(args.dir, "gold.json")
    log_path = os.path.join(args.dir, "retrieval_log.jsonl")
    csv_path = os.path.join(args.dir, "evaluation_results.csv")

    if not os.path.exists(gold_path):
        raise SystemExit(f"gold.json not found in {args.dir}")
    if not os.path.exists(log_path):
        raise SystemExit(
            f"retrieval_log.jsonl not found in {args.dir}.\n"
            f"Run the pipeline first:  python main.py  -> mode 2 or 3 on this question set."
        )

    gold_meta, gold = load_gold(gold_path)
    log = load_log(log_path)
    k = gold_meta.get("k", "?")
    min_sim = gold_meta.get("min_sim", "?")

    results = compute(gold, log)
    human = load_human_labels(csv_path) if os.path.exists(csv_path) else None

    print_report(results, human, k, min_sim, args.dir)

    if args.chart:
        if not HAVE_MPL:
            print(" Chart skipped: matplotlib not installed (pip install matplotlib).")
        else:
            out = args.chart_path or os.path.join(args.dir, "stress75_scorecard.png")
            make_chart(results, human, k, out)
            print(f" Chart written to {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
