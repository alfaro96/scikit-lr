# Reproduction of the papers

This directory reproduces the experiments of the papers whose methods scikit-lr implements, and compares the results of scikit-lr with the published ones. It is not part of the package or its tests: the experiments download data and take hours, so they run for each release on GitHub Actions, which opens a pull request with the results, and anyone can run them again on their own machine.

## Papers

* `cheng_decision_2009`: W. Cheng, J. Hühn and E. Hüllermeier, "Decision tree and instance-based learning for label ranking", in Proceedings of the 26th International Conference on Machine Learning, Montreal, Canada, pp. 161-168, 2009.
  * Methods: IBLR, `KNeighborsLabelRanker(model="mallows", weights="dudani")`.
  * Datasets: authorship, bodyfat, calhousing, cpu (cpu-small in the paper), elevators, fried, glass, housing, iris, pendigits, segment, stock, vehicle, vowel, wine and wisconsin, with the `label ranking` tag, in Table 1 (p. 6), whose numbers of samples are about 1.8 times those of OpenML and of `cheng_label_2010`, as the report shows.
  * Scenarios: complete training rankings, and 30% and 60% of the labels of each training ranking removed at random.
  * Metric: Kendall's $\tau$.
  * Cross-validation: five repetitions of a ten-fold cross-validation (p. 7), and an inner five-fold cross-validation, since the paper does not specify it.
  * Preprocessing: none, since the features are already normalized in OpenML, as the paper requires (p. 6).
  * Grid: the number of neighbors among 5, 10, 15 and 20, with the Euclidean distance.
  * Published results: Table 2 (p. 8).
  * Not specified: the grid of numbers of neighbors, so that of `cheng_label_2010` is used, the number of folds of the inner cross-validation, its score with incomplete rankings and the random splits.
* `cheng_label_2010`: W. Cheng, K. Dembczyński and E. Hüllermeier, "Label ranking methods based on the Plackett-Luce model", in Proceedings of the 27th International Conference on Machine Learning, Haifa, Israel, pp. 215-222, 2010.
  * Methods:
    * IB-PL, `KNeighborsLabelRanker(model="plackett_luce")`.
    * IB-Mal, `KNeighborsLabelRanker(model="mallows")`, without distance weights, as the paper does for a fair comparison with IB-PL.
  * Datasets: authorship, bodyfat, calhousing, cpu (cpu-small in the paper), elevators, fried, glass, housing, iris, pendigits, segment, stock, vehicle, vowel, wine and wisconsin, with the `label ranking` tag, in Table 1 (p. 6).
  * Scenarios: complete training rankings, and 30% and 60% of the labels of each training ranking removed at random.
  * Metric: Kendall's $\tau$.
  * Cross-validation: five repetitions of a ten-fold cross-validation (p. 6), and an inner five-fold cross-validation, since the paper does not specify it.
  * Preprocessing: none, since the features are already normalized in OpenML, as the paper requires (p. 5).
  * Grid: the number of neighbors among 5, 10, 15 and 20 (p. 5), with the Euclidean distance.
  * Published results: Table 2 (p. 7).
  * Not specified: the number of folds of the inner cross-validation, its score with incomplete rankings and the random splits.

Each directory holds the published scores (`paper_results.csv`, with the metric of each one) and datasets (`paper_datasets.csv`), with the table and page they come from, and the results of each version of scikit-lr under `results/<version>/`. `workflow/experiments.py` declares the datasets, scenarios, metric, cross-validation and methods of each paper, as its docstring describes, so adding a paper only takes its directory and its entry there.

## Datasets

The datasets are downloaded from OpenML, where the authors of scikit-lr uploaded them with the `label ranking` or the `partial label ranking` tag, and they are named as there. `OPENML_DATASETS` in `workflow/experiments.py` lists their identifiers under each tag, since a name may have several versions in OpenML. Each paper uses some of the datasets of a tag, and `paper_datasets.csv` gives the name of each one in the paper.

## Protocol

Each job runs one repetition of the outer cross-validation of a paper. In each fold, the training rankings are transformed as the scenario says, for instance removing some of their labels, the parameters of the method in its grid are chosen by the inner cross-validation of the paper, and the method is scored with the metric of the paper on the test rankings. The same seeds are used for every method and scenario, so they all get the same splits and random transformations. `workflow/run_cell.py` implements it.

The papers do not specify every detail of their protocol, as listed above, so the results are expected to be close to the published ones, not equal.

## Results

Each job writes `results/<version>/<dataset>/<method>/<scenario>/rep<repetition>.json` with the score, the chosen parameters, the number of convergence warnings, and the wall and CPU times of each fold, the parameters of the ranker, its grid, the cross-validation, the versions of the main packages and the machine where it ran (processor, cores, memory, system, Python, threading libraries and their threads), since the jobs of a run may run on different machines. Snakemake measures the time, memory and input and output of each job in the `.benchmark.tsv` file next to it. Each run also records the exact packages of the environment that writes the reports (`environment.txt`).

`results/<version>/report.md` compares the results with the published ones, and `results/<version>/results.csv` holds the aggregated results, with the mean and the standard deviation of the scores of all the folds of all the repetitions, which the example `examples/reproducibility/plot_paper_reproduction.py` of the documentation shows in interactive figures. The report lists the machines and the one of each result, since the times are only comparable between results of the same machine.

## Running the experiments

### On GitHub Actions

The "Reproducibility" workflow runs the experiments of a release, by hand from the branch with the commit of its version:

```
gh workflow run reproducibility.yml --ref <branch>
```

It downloads the datasets once, splits the experiments into batches that run in parallel, with `snakemake cells --config batch=<i>/<n>`, and then writes the reports and opens a pull request with the results against that branch. GitHub Actions must be allowed to create pull requests in the settings of the repository.

Adding the `reproducibility` label to a pull request runs the same experiments on its last commit, to check its changes, such as a new method, before a release. The reports are then stored as an artifact of the run instead of opening a pull request, and the label is removed, so that adding it again runs them again.

### On a machine of your own

The workflow and scikit-lr must come from the same commit, so check out the tag of the release to reproduce, or the commit of the development version, and from the root of the repository create the environment, which includes Snakemake, and install scikit-lr in it:

```
micromamba env create -f reproducibility/environment.yml
micromamba activate scikit-lr-reproducibility
pip install --no-build-isolation --no-deps .
```

The results of a development version are only for checking the changes before a release, and are not committed.

Then run the workflow from this directory, with the number of cores to use:

```
cd reproducibility
snakemake --cores 8
```

Snakemake only runs the jobs whose results are missing, so an interrupted run continues where it stopped (add `--rerun-incomplete` to repeat the jobs that were running), and the results of another version of scikit-lr go to their own directory. The experiments can be restricted with `--config`, for instance `snakemake --cores 8 --config 'datasets=["iris","wine"]' 'papers=["cheng_label_2010"]'`. The other options are `scenarios`, `batch` (the batch of the cells to run, such as `3/20`, with `snakemake cells`), `n_jobs` (the threads of each ranker, `1` by default) and `data_home` (the directory where the datasets are downloaded, `~/scikit_learn_data` by default).
