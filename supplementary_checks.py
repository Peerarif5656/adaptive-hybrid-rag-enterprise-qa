"""
supplementary_checks.py  -  F517783 primary-data companion to ahrag.py

Re-derives, from the stored results only (no API calls, no re-running of retrieval
or generation), the additional checks reported in the final dissertation:

  1. Refusal rates and how the RAGAS judge scored refusals            (Section 3.6)
  2. Faithfulness on questions answered in both configurations         (Section 4.3)
  3. nDCG@5 recomputed with each gold document counted once            (Sections 3.6, 4.4)
  4. Composition of the error-analysis categories (refusal vs answer)  (Section 4.6)
  5. Top-5 passage overlap between the fixed-weight and adaptive hybrid (Section 5.2)
  6. Router behaviour: weight vs query length, extremes, Azure labels  (Sections 3.3, 4.5, 5.4)
  7. Dense vs static-hybrid statistics, coverage and relevancy         (Sections 4.2, 4.3)
  8. Approximate share of passages longer than the dense encoder limit (Section 6.3)

Usage (from the archive root):
    python supplementary_checks.py
Output is printed and also written to results/supplementary_checks.txt.
Dependencies: numpy, pandas, scipy (already pinned in requirements.txt).
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
REFUSAL = "i don't have enough information"
SEED = 42
CONFIGS = [
    "sparse_bm25", "dense", "static_hybrid", "static_hybrid_cite",
    "static_hybrid_weighted", "static_hybrid_weighted_cite",
    "adaptive_hybrid", "adaptive_hybrid_cite",
]


# --------------------------------------------------------------------------- helpers
def load_audit(dataset: str, config: str) -> list[dict]:
    path = RESULTS / dataset / "audit" / f"{config}.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10_000, seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean difference (same scheme as ahrag.py)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(diff: np.ndarray) -> float:
    if diff.size == 0 or np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values with monotonicity (same as ahrag.py)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted.tolist()


def ndcg_dedup(record: dict, k: int = 5) -> float:
    """nDCG@k counting each gold document once (first occurrence only)."""
    gold = set(record["gold_doc_ids"])
    if not gold:
        return math.nan
    seen: set[str] = set()
    dcg = 0.0
    for i, r in enumerate(record["retrieved"][:k]):
        if r["doc_id"] in gold and r["doc_id"] not in seen:
            seen.add(r["doc_id"])
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else math.nan


def header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# --------------------------------------------------------------------------- analyses
def section_1_refusals(df: pd.DataFrame) -> None:
    header("1. Refusal rates and judge scoring of refusals (Section 3.6)")
    rates = (df.groupby(["dataset", "config"])["refusal"].mean() * 100).round(1).unstack(0)
    print("Refusal rate (%) by configuration and dataset:")
    print(rates.reindex(CONFIGS).to_string())
    ref = df[df["refusal"]]
    print(f"\nTotal refusals: {len(ref)}")
    print(f"  scored faithfulness = 1.0 : {(ref['faithfulness'] == 1.0).sum()}")
    print(f"  scored faithfulness = 0.0 : {(ref['faithfulness'] == 0.0).sum()}")
    print(f"  other values / NaN        : {len(ref) - (ref['faithfulness'] == 1.0).sum() - (ref['faithfulness'] == 0.0).sum()}")
    for ds in ["financebench", "cuad"]:
        r = ref[ref.dataset == ds]["faithfulness"]
        print(f"  {ds}: share scored 1.0 = {(r == 1.0).mean():.1%}, share scored 0.0 = {(r == 0.0).mean():.1%}")
    bm = df[(df.dataset == "financebench") & (df.config == "sparse_bm25")]
    print("\nBM25 on FinanceBench: mean faithfulness")
    print(f"  all questions     : {bm['faithfulness'].mean():.3f} (n = {len(bm)})")
    print(f"  answered questions: {bm.loc[~bm.refusal, 'faithfulness'].mean():.3f} (n = {(~bm.refusal).sum()})")
    print(f"  refusals          : {bm.loc[bm.refusal, 'faithfulness'].mean():.3f} (n = {bm.refusal.sum()})")
    cu = df[(df.dataset == "cuad") & (df["recall@5"] == 0) & (df.refusal)]
    print(f"\nCUAD refusals with Recall@5 = 0: n = {len(cu)}, scored 0.0 = {(cu['faithfulness'] == 0.0).sum()}")


def section_2_answered_only(df: pd.DataFrame) -> None:
    header("2. Faithfulness on questions answered (non-refusal) in BOTH configurations (Section 4.3) - exploratory")
    pairs = [
        ("RQ2", "static_hybrid", "static_hybrid_cite"),
        ("RQ2", "adaptive_hybrid", "adaptive_hybrid_cite"),
        ("extra", "static_hybrid_weighted", "static_hybrid_weighted_cite"),
        ("RQ3", "static_hybrid_weighted", "adaptive_hybrid"),
    ]
    for ds in ["financebench", "cuad"]:
        sub = df[df.dataset == ds]
        for rq, a_cfg, b_cfg in pairs:
            a = sub[sub.config == a_cfg].set_index("qid")
            b = sub[sub.config == b_cfg].set_index("qid")
            q = a.index.intersection(b.index)
            keep = (~a.loc[q, "refusal"]) & (~b.loc[q, "refusal"]) \
                & a.loc[q, "faithfulness"].notna() & b.loc[q, "faithfulness"].notna()
            qa = keep[keep].index
            diff = (b.loc[qa, "faithfulness"] - a.loc[qa, "faithfulness"]).to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff)
            dz = diff.mean() / diff.std(ddof=1) if diff.size > 1 and diff.std(ddof=1) > 0 else 0.0
            print(f"{ds:12s} {rq:5s} {a_cfg:27s} -> {b_cfg:27s} n={diff.size:3d} "
                  f"mean_a={a.loc[qa, 'faithfulness'].mean():.3f} mean_b={b.loc[qa, 'faithfulness'].mean():.3f} "
                  f"diff={diff.mean():+.3f} CI[{lo:+.3f}, {hi:+.3f}] dz={dz:+.2f} raw_p={wilcoxon_p(diff):.5f}")


def section_3_ndcg(stats_df: pd.DataFrame) -> None:
    header("3. nDCG@5 recomputed with each gold document counted once (Sections 3.6, 4.4)")
    all_results = pd.read_csv(RESULTS / "all_results.csv", low_memory=False)
    over = all_results[all_results["ndcg@5"] > 1.0 + 1e-9].groupby(["dataset", "config"]).size()
    print("Per-question nDCG@5 values above 1.0 in the stored results (chunk-level crediting):")
    print(over.to_string() if len(over) else "  none")
    print(f"Maximum stored nDCG@5: {all_results['ndcg@5'].max():.3f}")

    comparisons = [
        ("sparse_bm25", "static_hybrid"),
        ("dense", "static_hybrid"),
        ("static_hybrid_weighted", "adaptive_hybrid"),
        ("static_hybrid_weighted_cite", "adaptive_hybrid_cite"),
    ]
    for ds in ["financebench", "cuad", "azure"]:
        vals = {c: pd.Series({r["qid"]: ndcg_dedup(r) for r in load_audit(ds, c)}) for c in CONFIGS}
        print(f"\n{ds}: de-duplicated nDCG@5 means: " +
              ", ".join(f"{c}={vals[c].mean():.3f}" for c in CONFIGS))
        new_p = {}
        for a_cfg, b_cfg in comparisons:
            a, b = vals[a_cfg], vals[b_cfg]
            q = a.index.intersection(b.index)
            diff = (b[q] - a[q]).to_numpy(dtype=float)
            p = wilcoxon_p(diff)
            new_p[(a_cfg, b_cfg)] = p
            lo, hi = bootstrap_ci(diff)
            dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
            print(f"  {a_cfg:27s} -> {b_cfg:22s} diff={diff.mean():+.3f} CI[{lo:+.3f}, {hi:+.3f}] "
                  f"dz={dz:+.3f} raw_p={p:.2e}")
        fam = stats_df[stats_df.dataset == ds].reset_index(drop=True)
        pvals = []
        for _, row in fam.iterrows():
            key = (row.config_a, row.config_b)
            pvals.append(new_p[key] if row.metric == "ndcg@5" and key in new_p else row.wilcoxon_p)
        adjusted = holm(pvals)
        for i, row in fam.iterrows():
            if row.metric == "ndcg@5" and (row.config_a, row.config_b) in new_p:
                print(f"  Holm-adjusted p (32-test family, de-duplicated nDCG@5) "
                      f"{row.config_a} -> {row.config_b}: {adjusted[i]:.2e}")


def section_4_error_composition(df: pd.DataFrame) -> None:
    header("4. Error-analysis categories split by refusal vs answer (Section 4.6)")
    ea = pd.read_csv(RESULTS / "error_analysis.csv", low_memory=False)
    ea["refusal"] = ea["answer"].fillna("").str.lower().str.contains(REFUSAL, regex=False)
    table = ea.groupby(["dataset", "config", "error_category", "refusal"]).size().unstack(fill_value=0)
    table.columns = ["answered" if not c else "refusal" for c in table.columns]
    print(table.to_string())


def section_5_overlap() -> None:
    header("5. Top-5 passage overlap: fixed-weight vs adaptive hybrid (Section 5.2)")
    for ds in ["financebench", "cuad", "azure"]:
        fixed = {r["qid"]: [x["chunk_id"] for x in r["retrieved"]] for r in load_audit(ds, "static_hybrid_weighted")}
        adapt = {r["qid"]: [x["chunk_id"] for x in r["retrieved"]] for r in load_audit(ds, "adaptive_hybrid")}
        shared = [len(set(fixed[q]) & set(adapt[q])) for q in fixed]
        jacc = [len(set(fixed[q]) & set(adapt[q])) / len(set(fixed[q]) | set(adapt[q])) for q in fixed]
        same_set = np.mean([set(fixed[q]) == set(adapt[q]) for q in fixed])
        same_order = np.mean([fixed[q] == adapt[q] for q in fixed])
        print(f"{ds:12s} mean shared passages={np.mean(shared):.2f}/5, mean Jaccard={np.mean(jacc):.3f}, "
              f"identical set={same_set:.1%}, identical order={same_order:.1%}")


def section_6_router() -> None:
    header("6. Router behaviour (Sections 3.3, 4.5, 5.4)")
    for ds in ["financebench", "cuad", "azure"]:
        recs = load_audit(ds, "adaptive_hybrid")
        w = np.array([r["routing"]["dense_weight"] for r in recs])
        n = np.array([r["routing"]["n_tokens"] for r in recs])
        rho, p = stats.spearmanr(n, w)
        labels = pd.Series([r["routing"]["label"] for r in recs]).value_counts().to_dict()
        print(f"{ds:12s} weight min={w.min():.3f} max={w.max():.3f} mean={w.mean():.3f} SD={w.std(ddof=1):.3f} "
              f"| >0.7: {(w > 0.7).sum()} <0.3: {(w < 0.3).sum()} | Spearman(n_tokens, weight)={rho:.2f} "
              f"(p={p:.1e}) | labels={labels}")
        if ds == "financebench":
            ordered = sorted(recs, key=lambda r: r["routing"]["dense_weight"])
            for r in ordered[:1] + ordered[-1:]:
                print(f"   extreme: w={r['routing']['dense_weight']:.3f} label={r['routing']['label']} "
                      f"n_tokens={r['routing']['n_tokens']} | {r['question'][:110]}")
            capex = next(r for r in recs if r["qid"] == "financebench_id_03029")
            print(f"   3M FY2018 capex question routing: {capex['routing']}")
    qa = {json.loads(l)["qid"]: json.loads(l) for l in open(DATA / "azure" / "qa.jsonl", encoding="utf-8")}
    recs = {r["qid"]: r for r in load_audit("azure", "adaptive_hybrid")}
    author = pd.Series([qa[q]["metadata"]["qtype"] for q in qa], name="author_label")
    router = pd.Series([recs[q]["routing"]["label"] for q in qa], name="router_label")
    print("\nAzure: author-assigned query type vs router label")
    print(pd.crosstab(author, router).to_string())


def section_7_descriptives(df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    header("7. Dense vs static hybrid (RQ1), citation coverage and answer relevancy (Sections 4.2, 4.3)")
    sub = stats_df[(stats_df.dataset == "financebench") & (stats_df.config_a == "dense")]
    print(sub[["metric", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
               "cohens_dz", "wilcoxon_p", "p_adjusted"]].round(4).to_string(index=False))
    summary = pd.read_csv(RESULTS / "summary.csv")
    print("\nCitation coverage (non-refusal answers) and answer relevancy:")
    print(summary[["dataset", "config", "citation_coverage", "answer_relevancy"]].round(3).to_string(index=False))
    q3m = df[(df.dataset == "financebench") & (df.qid == "financebench_id_00499")]
    print("\n'Is 3M capital-intensive (FY2022)?' faithfulness by configuration:")
    for _, row in q3m.iterrows():
        print(f"   {row.config:28s} faithfulness={row.faithfulness:.2f} refusal={row.refusal}")


def section_8_truncation() -> None:
    header("8. Passages longer than the dense encoder limit of 256 word pieces (Section 6.3) - approximate")
    try:
        from transformers import AutoTokenizer  # exact count when available locally
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        count = lambda text: len(tok(text, add_special_tokens=True)["input_ids"])
        method = "exact (MiniLM tokenizer)"
    except Exception:
        def count(text: str) -> int:
            n = 0
            for t in re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", text):
                if t.isdigit():
                    n += max(1, math.ceil(len(t) / 2))
                elif t.isalpha():
                    n += 1 if len(t) <= 8 else 2
                else:
                    n += 1
            return n
        method = "approximate (regex word-piece estimate; tokenizer not available offline)"
    docs = [json.loads(l) for l in open(DATA / "financebench" / "corpus.jsonl", encoding="utf-8")]
    lengths = np.array([count(d["text"]) for d in docs])
    print(f"Method: {method}")
    print(f"FinanceBench evidence passages: median length {int(np.median(lengths))}, "
          f"share above 256 = {np.mean(lengths > 256):.0%}")


def main() -> int:
    df = pd.read_csv(RESULTS / "all_results.csv", low_memory=False)
    df["refusal"] = df["answer"].fillna("").str.lower().str.contains(REFUSAL, regex=False)
    stats_df = pd.read_csv(RESULTS / "stats.csv")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        section_1_refusals(df)
        section_2_answered_only(df)
        section_3_ndcg(stats_df)
        section_4_error_composition(df)
        section_5_overlap()
        section_6_router()
        section_7_descriptives(df, stats_df)
        section_8_truncation()
    text = buffer.getvalue()
    print(text)
    (RESULTS / "supplementary_checks.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
