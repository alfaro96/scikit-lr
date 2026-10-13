"""
==========================
Reproduction of the papers
==========================

This example compares the scores of the methods of scikit-lr with those
published in the papers that propose them, with the metric, the datasets and the
scenarios of each paper, such as training rankings with some labels removed. Each
paper is named by its citation key in the user guide.

The experiments take hours, so they are not run here. They are run with the
workflow of the ``reproducibility`` directory of the repository of scikit-lr, whose
``README.md`` lists the papers with their methods and protocol, and this example
reads the results committed there for the latest version of scikit-lr that has
them. The figures are interactive: hovering over a mark shows its dataset and
values.
"""

# %%
# Results
# -------
#
# The results of each paper are in ``results/<version>/results.csv``, with the
# mean and the standard deviation of the score over the folds of all the
# repetitions, and the published ones in ``paper_results.csv``.

import csv
import itertools
import re
from pathlib import Path

import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots


def find_reproducibility():
    """Find the ``reproducibility`` directory of the repository.

    It is searched from the working directory upwards, so that the example runs
    from any directory of the repository.
    """
    for directory in [Path.cwd(), *Path.cwd().parents]:
        if (directory / "reproducibility" / "Snakefile").exists():
            return directory / "reproducibility"
    raise FileNotFoundError("Run this example from the repository of scikit-lr.")


REPRODUCIBILITY = find_reproducibility()
# Each directory with the published results of a paper is named by its citation key
PAPERS = sorted(
    path.parent.name for path in REPRODUCIBILITY.glob("*/paper_results.csv")
)


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def version_key(path):
    """Order the directories of the versions by the numbers of their names."""
    return [int(number) for number in re.findall(r"\d+", path.parent.name)]


def load_results(paper):
    """Join the results of the latest version with those published in the paper.

    Returns the version and a row for each dataset, method and scenario, or
    ``None`` and no rows if no version has results. The published score is
    ``None`` if the paper does not give it.
    """
    paths = sorted((REPRODUCIBILITY / paper / "results").glob("*/results.csv"))
    if not paths:
        return None, []
    path = max(paths, key=version_key)
    published = {
        (row["dataset"], row["method"], row["scenario"]): float(row["score"])
        for row in read_csv(REPRODUCIBILITY / paper / "paper_results.csv")
    }
    rows = []
    for row in read_csv(path):
        key = (row["dataset"], row["method"], row["scenario"])
        rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "scenario": row["scenario"],
                "metric": row["metric_latex"],
                "paper": published.get(key),
                "mean": float(row["score_mean"]),
                "std": float(row["score_std"]),
                "best_params": row["best_params_mode"],
                "wall_time": sum(
                    float(row[step])
                    for step in ["search_wall", "fit_wall", "predict_wall"]
                ),
            }
        )
    return path.parent.name, rows


results = {}
for paper in PAPERS:
    version, rows = load_results(paper)
    if not rows:
        print(f"{paper}: no results yet")
    else:
        print(f"{paper}: results of scikit-lr {version}")
        results[paper] = rows

# Each method keeps its color in every figure
methods = sorted({row["method"] for rows in results.values() for row in rows})
COLORS = dict(zip(methods, itertools.cycle(qualitative.Plotly)))


def select(rows, published=False, **values):
    """Select the rows with the given values, and with a published score if asked."""
    return [
        row
        for row in rows
        if all(row[key] == value for key, value in values.items())
        and (not published or row["paper"] is not None)
    ]


def scenarios(paper):
    """List the scenarios of a paper in the order of its results."""
    return list(dict.fromkeys(row["scenario"] for row in results[paper]))


def metric(paper):
    """Return the metric of a paper in LaTeX, which Plotly renders with MathJax."""
    return results[paper][0]["metric"]


def scenario_titles(paper, n_cols):
    titles = [f"{paper}, {scenario}" for scenario in scenarios(paper)]
    return titles + [""] * (n_cols - len(titles))


def max_scenarios():
    return max(len(scenarios(paper)) for paper in results)


# %%
# Published and obtained scores
# -----------------------------
#
# Each mark is a dataset, with the published score on the horizontal axis and that
# of scikit-lr on the vertical one, with its standard deviation, and each row of
# panels is a paper. The marks on the diagonal reproduce the published value. The
# papers do not specify their random splits or every detail of their protocol, so
# the results are expected to be close to the published ones, not equal.


def plot_published_and_obtained():
    if not results:
        return None
    n_cols = max_scenarios()
    fig = make_subplots(
        rows=len(results),
        cols=n_cols,
        subplot_titles=[t for paper in results for t in scenario_titles(paper, n_cols)],
    )
    shown = set()
    for i, (paper, rows) in enumerate(results.items(), start=1):
        for col, scenario in enumerate(scenarios(paper), start=1):
            for method in sorted({row["method"] for row in rows}):
                selected = select(rows, True, method=method, scenario=scenario)
                fig.add_trace(
                    go.Scatter(
                        x=[row["paper"] for row in selected],
                        y=[row["mean"] for row in selected],
                        error_y={"array": [row["std"] for row in selected]},
                        mode="markers",
                        marker={"color": COLORS[method]},
                        name=method,
                        legendgroup=method,
                        showlegend=method not in shown,
                        text=[row["dataset"] for row in selected],
                        customdata=[
                            [row["mean"] - row["paper"], row["best_params"]]
                            for row in selected
                        ],
                        hovertemplate=(
                            "%{text}<br>paper: %{x:.3f}<br>scikit-lr: %{y:.3f}"
                            "<br>Δ: %{customdata[0]:+.3f}"
                            "<br>parameters: %{customdata[1]}"
                        ),
                    ),
                    row=i,
                    col=col,
                )
                shown.add(method)
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line={"color": "gray", "dash": "dash"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=i,
                col=col,
            )
            fig.update_xaxes(
                title_text=f"$\\text{{Paper }}({metric(paper)})$", row=i, col=col
            )
        fig.update_yaxes(
            title_text=f"$\\text{{scikit-lr }}({metric(paper)})$", row=i, col=1
        )
    fig.update_layout(title="Published and obtained scores", height=400 * len(results))
    return fig


plot_published_and_obtained()

# %%
# Difference with the papers
# --------------------------
#
# The difference between the score of scikit-lr and the published one, for each
# dataset and method, shows which datasets are farther from the papers.


def plot_differences():
    if not results:
        return None
    titles = [title for paper in results for title in scenario_titles(paper, 0)]
    fig = make_subplots(rows=len(titles), cols=1, subplot_titles=titles)
    shown = set()
    row_index = 0
    for paper, rows in results.items():
        for scenario in scenarios(paper):
            row_index += 1
            for method in sorted({row["method"] for row in rows}):
                selected = select(rows, True, method=method, scenario=scenario)
                fig.add_trace(
                    go.Bar(
                        x=[row["dataset"] for row in selected],
                        y=[row["mean"] - row["paper"] for row in selected],
                        marker={"color": COLORS[method]},
                        name=method,
                        legendgroup=method,
                        showlegend=method not in shown,
                        hovertemplate="%{x}<br>Δ: %{y:+.3f}",
                    ),
                    row=row_index,
                    col=1,
                )
                shown.add(method)
            fig.update_yaxes(
                title_text=f"$\\Delta\\, {metric(paper)}$", row=row_index, col=1
            )
    fig.update_layout(
        title="scikit-lr minus paper", barmode="group", height=250 * row_index
    )
    return fig


plot_differences()

# %%
# Scenarios
# ---------
#
# The scores of each dataset across the scenarios of a paper, such as removing
# more and more labels from the training rankings. Each panel is a method of a
# paper and each line a dataset, with the result of scikit-lr as a solid line and
# the published one as a dotted line of the same color.


def plot_scenarios():
    if not results:
        return None
    paper_methods = {
        paper: sorted({row["method"] for row in rows})
        for paper, rows in results.items()
    }
    n_cols = max(len(methods) for methods in paper_methods.values())
    titles = []
    for paper, methods in paper_methods.items():
        titles += [f"{paper}, {method}" for method in methods]
        titles += [""] * (n_cols - len(methods))
    fig = make_subplots(rows=len(results), cols=n_cols, subplot_titles=titles)
    shown = set()
    for i, (paper, rows) in enumerate(results.items(), start=1):
        datasets = sorted({row["dataset"] for row in rows})
        paper_scenarios = scenarios(paper)
        for col, method in enumerate(paper_methods[paper], start=1):
            for j, dataset in enumerate(datasets):
                by_scenario = {
                    row["scenario"]: row
                    for row in select(rows, method=method, dataset=dataset)
                }
                if len(by_scenario) < len(paper_scenarios):
                    continue
                color = f"hsl({360 * j / len(datasets):.0f}, 60%, 45%)"
                for key, dash, label in [
                    ("mean", "solid", "scikit-lr"),
                    ("paper", "dot", "paper"),
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=paper_scenarios,
                            y=[
                                by_scenario[scenario][key]
                                for scenario in paper_scenarios
                            ],
                            mode="lines+markers",
                            line={"color": color, "dash": dash},
                            name=dataset,
                            legendgroup=dataset,
                            showlegend=key == "mean" and dataset not in shown,
                            hovertemplate=f"{dataset} ({label})<br>%{{y:.3f}}",
                        ),
                        row=i,
                        col=col,
                    )
                shown.add(dataset)
        fig.update_yaxes(title_text=f"${metric(paper)}$", row=i, col=1)
    fig.update_layout(title="Scenarios", height=400 * len(results))
    return fig


plot_scenarios()

# %%
# Time
# ----
#
# The wall time of all the repetitions of the cross-validation of each dataset,
# including the choice of the parameters, on the machine described in the report
# of each version. The times are only comparable between runs on the same machine.


def plot_times():
    if not results:
        return None
    n_cols = max_scenarios()
    fig = make_subplots(
        rows=len(results),
        cols=n_cols,
        shared_yaxes=True,
        subplot_titles=[t for paper in results for t in scenario_titles(paper, n_cols)],
    )
    shown = set()
    for i, (paper, rows) in enumerate(results.items(), start=1):
        for col, scenario in enumerate(scenarios(paper), start=1):
            for method in sorted({row["method"] for row in rows}):
                selected = select(rows, method=method, scenario=scenario)
                fig.add_trace(
                    go.Bar(
                        x=[row["dataset"] for row in selected],
                        y=[row["wall_time"] for row in selected],
                        marker={"color": COLORS[method]},
                        name=method,
                        legendgroup=method,
                        showlegend=method not in shown,
                        hovertemplate="%{x}<br>%{y:.1f} s",
                    ),
                    row=i,
                    col=col,
                )
                shown.add(method)
        fig.update_yaxes(title_text="Wall time (s)", type="log", row=i, col=1)
    fig.update_layout(title="Time", barmode="group", height=400 * len(results))
    return fig


plot_times()
