"""Compare the results of scikit-lr with those published in a paper.

Reads the results of every repetition of a version of scikit-lr, with the machine
where each one ran, the published results and datasets of the paper and the
measurements of Snakemake, and writes the aggregated results as CSV and a report in
Markdown.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from experiments import METRICS, PAPERS

# The columns of the aggregated results, which are written even without results
COLUMNS = [
    "dataset",
    "method",
    "scenario",
    "metric",
    "metric_latex",
    "n_repetitions",
    "n_folds",
    "n_samples",
    "score_mean",
    "score_std",
    "best_params_mode",
    "convergence_warnings",
    "search_wall",
    "fit_wall",
    "predict_wall",
    "cpu",
    "max_rss_mb",
    "machines",
]


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file, skipinitialspace=True))


def read_benchmark(path):
    """Read the measurements of a job by Snakemake, a row of a tab separated file."""
    with open(path, newline="") as file:
        return next(csv.DictReader(file, delimiter="\t"))


def aggregate(paper, results_dir):
    """Aggregate the folds and repetitions of each dataset, method and scenario.

    Returns the rows, in the order of the scenarios of the paper, and the distinct
    machines where the repetitions ran, which each row names by their position in
    that list, from ``M1``.
    """
    cells = defaultdict(list)
    machines = []
    for path in sorted(results_dir.glob("*/*/*/rep*.json")):
        result = json.loads(path.read_text())
        key = (result["dataset"], result["method"], result["scenario"])
        cells[key].append((result, path.with_suffix(".benchmark.tsv")))
        if result["machine"] not in machines:
            machines.append(result["machine"])

    rows = []
    scenarios = list(PAPERS[paper]["scenarios"])
    order = sorted(cells, key=lambda key: (key[0], key[1], scenarios.index(key[2])))
    for dataset, method, scenario in order:
        repetitions = cells[dataset, method, scenario]
        folds = [fold for result, _ in repetitions for fold in result["folds"]]
        scores = np.array([fold["score"] for fold in folds])
        best_params = Counter(
            ", ".join(
                f"{name}={value}" for name, value in sorted(f["best_params"].items())
            )
            for f in folds
        )
        benchmarks = [read_benchmark(path) for _, path in repetitions if path.exists()]
        row_machines = sorted(
            {f"M{machines.index(result['machine']) + 1}" for result, _ in repetitions}
        )
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "scenario": scenario,
                "metric": PAPERS[paper]["metric"],
                "metric_latex": METRICS[PAPERS[paper]["metric"]]["latex"],
                "n_repetitions": len(repetitions),
                "n_folds": len(folds),
                "n_samples": repetitions[0][0]["n_samples"],
                "score_mean": scores.mean(),
                "score_std": scores.std(ddof=1) if len(scores) > 1 else 0.0,
                "best_params_mode": best_params.most_common(1)[0][0],
                "convergence_warnings": sum(f["convergence_warnings"] for f in folds),
                "search_wall": sum(f["time"]["search_wall"] for f in folds),
                "fit_wall": sum(f["time"]["fit_wall"] for f in folds),
                "predict_wall": sum(f["time"]["predict_wall"] for f in folds),
                "cpu": sum(
                    f["time"][f"{step}_cpu"]
                    for f in folds
                    for step in ["search", "fit", "predict"]
                ),
                "machines": ", ".join(row_machines),
                # Snakemake writes NA for what it could not measure
                "max_rss_mb": max(
                    (float(b["max_rss"]) for b in benchmarks if b["max_rss"] != "NA"),
                    default=np.nan,
                ),
            }
        )
    return rows, machines


def write_csv(rows, path):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def report(paper, version, rows, machines, paper_results, paper_datasets):
    published = {
        (r["dataset"], r["method"], r["scenario"]): float(r["score"])
        for r in paper_results
    }
    sources = sorted({r["source"] for r in paper_results})
    ours = {(r["dataset"], r["method"], r["scenario"]): r for r in rows}
    methods = list(PAPERS[paper]["methods"])
    scenarios = PAPERS[paper]["scenarios"]
    metric = f"${METRICS[PAPERS[paper]['metric']]['latex']}$"

    introduction = (
        f"Generated by `reproducibility/workflow/report.py`. {metric} of each "
        "method, as published and as obtained with scikit-lr (mean ± standard "
        "deviation over the folds of all the repetitions), and their difference "
        f"Δ = scikit-lr - paper. The published values come from {', '.join(sources)} "
        "of the paper. The protocol is described in `reproducibility/README.md`."
    )
    lines = [
        f"# Reproduction of {paper} with scikit-lr {version}",
        "",
        introduction,
        "",
        "## Machines",
        "",
        (
            "The machines where the repetitions ran, which the tables of the "
            "scenarios name. The times are only comparable between results of the "
            "same machine."
        ),
        "",
        markdown_table(
            [
                "machine",
                "processor",
                "cores (physical, logical)",
                "memory (GiB)",
                "system",
                "Python",
                "threading libraries (threads)",
            ],
            [
                [
                    f"M{i}",
                    machine["processor"],
                    f"{machine['physical_cores']}, {machine['logical_cores']}",
                    f"{machine['memory_bytes'] / 2**30:.1f}",
                    machine["system"],
                    machine["python"],
                    ", ".join(
                        f"{pool['internal_api']} ({pool['num_threads']})"
                        for pool in machine["threadpools"]
                    ),
                ]
                for i, machine in enumerate(machines, start=1)
            ],
        ),
        "",
        "## Datasets",
        "",
        markdown_table(
            [
                "dataset (OpenML)",
                "name in the paper",
                "samples (paper)",
                "samples (OpenML)",
            ],
            [
                [
                    d["dataset"],
                    d["paper_name"],
                    d["n_samples"],
                    next(
                        (r["n_samples"] for r in rows if r["dataset"] == d["dataset"]),
                        "—",
                    ),
                ]
                for d in paper_datasets
            ],
        ),
    ]
    for scenario in scenarios:
        table = []
        for d in paper_datasets:
            for method in methods:
                key = (d["dataset"], method, scenario)
                # The paper may not publish every score, and the results may be
                # restricted to some datasets and scenarios
                paper_score = published.get(key)
                paper_cell = "—" if paper_score is None else f"{paper_score:.3f}"
                row = ours.get(key)
                if row is None:
                    table.append([d["dataset"], method, paper_cell] + ["—"] * 7)
                    continue
                mean, std = row["score_mean"], row["score_std"]
                difference = (
                    "—" if paper_score is None else f"{mean - paper_score:+.3f}"
                )
                wall = row["search_wall"] + row["fit_wall"] + row["predict_wall"]
                table.append(
                    [
                        d["dataset"],
                        method,
                        paper_cell,
                        f"{mean:.3f} ± {std:.3f}",
                        difference,
                        row["best_params_mode"],
                        row["convergence_warnings"],
                        f"{wall:.1f}",
                        f"{row['cpu']:.1f}",
                        row["machines"],
                    ]
                )
        lines += [
            "",
            f"## Scenario: {scenario} ({scenarios[scenario]['description']})",
            "",
            markdown_table(
                [
                    "dataset",
                    "method",
                    "paper",
                    "scikit-lr",
                    "Δ",
                    "chosen parameters (mode)",
                    "convergence warnings",
                    "wall time (s)",
                    "CPU time (s)",
                    "machine",
                ],
                table,
            ),
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    paper_dir = Path(__file__).parents[1] / args.paper
    rows, machines = aggregate(args.paper, args.results_dir)
    write_csv(rows, args.csv)
    args.report.write_text(
        report(
            args.paper,
            args.version,
            rows,
            machines,
            read_csv(paper_dir / "paper_results.csv"),
            read_csv(paper_dir / "paper_datasets.csv"),
        )
    )


if __name__ == "__main__":
    main()
