# score_eval.py
# Compare a stress-test run against the gold answer key, print the numbers, and draw a chart.
#
# It reads three files from an evaluation folder (default: evaluation/stress_75):
#   gold.json               the answer key: which FAQ each question should retrieve
#   retrieval_log.jsonl     what the pipeline actually retrieved (written by the eval modes)
#   evaluation_results.csv  your manual labels (optional, for the answer-quality numbers)
#
# From the log it works out: retrieval recall@k for the in-scope questions (overall and per
# category), how often multi-part questions retrieved both expected FAQs, and how often
# out-of-scope and adversarial questions were declined.
#
# Run:
#   python evaluation/score_eval.py
#   python evaluation/score_eval.py --dir evaluation/stress_75
#   python evaluation/score_eval.py --no-chart
#
# Matching is case-insensitive: an expected FAQ counts as retrieved if its key phrase shows
# up in any of the retrieved FAQ questions.
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # render to a file, no window
import matplotlib.pyplot as plt

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
    # work out all the numbers from the gold key and the retrieval log
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

        # out-of-scope and adversarial questions have no correct FAQ; the right behaviour is
        # to decline, so we just track whether retrieval was refused
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
        # "all" means every expected FAQ has to show up (multi-part); "any" means one is enough
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
        print(f"\n{len(m)} questions in the gold file had no log entry (idx {m[:10]}"
              f"{'...' if len(m) > 10 else ''}). Re-run over the full set.")

    print("\nRetrieval (worked out from the log):")
    print(f"  In-scope recall@{k}: {pct((r['inscope_hit'], r['inscope_total']))}")
    print("  By category:")
    for cat, ht in r["categories"].items():
        print(f"    {cat:<14} {pct(ht)}")
    if r["multi_total"]:
        print(f"  Multi-part, both FAQs retrieved: {pct((r['multi_both'], r['multi_total']))}; "
              f"at least one: {pct((r['multi_either'], r['multi_total']))}")

    print("\nScope and safety (did it decline to retrieve?):")
    print(f"  Out-of-scope refusal: {pct((r['oos_refused'], r['oos_total']))}")
    print(f"  Adversarial refusal:  {pct((r['adv_refused'], r['adv_total']))}")
    print("  This only checks whether retrieval was suppressed. Whether the answer itself")
    print("  safely declined is your call (the no_hallucination_on_oos column).")

    if human:
        print("\nYour manual labels (from evaluation_results.csv):")
        for c, ht in human.items():
            print(f"  {c:<24} {pct(ht)}")
    else:
        print("\nNo manual labels found yet (run the labeling mode to add them).")
    print("\n" + "=" * 64)


def make_chart(r, human, k, out_path):
    # draw the scorecard: the three labeled metrics plus the recall@k number. if there are no
    # labels yet (a retrieval-only run) we show the automatic numbers instead.
    bars = []  # (label, value_pct, color)
    n_label = ""
    if human:
        order = [
            ("Retrieval success", "retrieval_success", "#4C72B0"),
            ("Answer correctness", "answer_correct", "#55A868"),
            ("No hallucination on OOS", "no_hallucination_on_oos", "#C44E52"),
        ]
        for label, key, color in order:
            if key in human:
                hit, total = human[key]
                bars.append((label, 100 * hit / total, color))
                n_label = f" (N={total})"
        if r["inscope_total"]:
            bars.append((f"Retrieval recall@{k}",
                         100 * r["inscope_hit"] / r["inscope_total"], "#8172B3"))
    else:
        # no labels yet: show the automatic retrieval and refusal numbers
        def add(label, ht, color):
            hit, total = ht
            if total:
                bars.append((label, 100 * hit / total, color))
        add(f"Retrieval recall@{k}", (r["inscope_hit"], r["inscope_total"]), "#4C72B0")
        add("Hard-negative recall", r["categories"].get("hard_negative", (0, 0)), "#55A868")
        add("OOS refusal", (r["oos_refused"], r["oos_total"]), "#C44E52")

    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 2.1), 5))
    xb = ax.bar(range(len(labels)), values, color=colors, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 110)  # a little room above 100 so a full bar's label clears the title
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"RAG Chatbot Evaluation on FAQ Dataset{n_label}", pad=12)
    for b, v in zip(xb, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="evaluation/stress_75")
    ap.add_argument("--no-chart", dest="chart", action="store_false")
    ap.set_defaults(chart=True)
    args = ap.parse_args()

    gold_meta, gold = load_gold(os.path.join(args.dir, "gold.json"))
    log = load_log(os.path.join(args.dir, "retrieval_log.jsonl"))
    k = gold_meta.get("k", "?")
    min_sim = gold_meta.get("min_sim", "?")

    results = compute(gold, log)
    csv_path = os.path.join(args.dir, "evaluation_results.csv")
    human = load_human_labels(csv_path) if os.path.exists(csv_path) else None

    print_report(results, human, k, min_sim, args.dir)

    if args.chart:
        out = os.path.join(args.dir, "stress75_scorecard.png")
        make_chart(results, human, k, out)
        print(f" Chart written to {out}")


if __name__ == "__main__":
    main()
