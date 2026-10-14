# Reproduction of the papers

This directory reproduces the experiments of the papers whose methods scikit-lr implements, and compares the results of scikit-lr with the published ones. It is not part of the package or its tests: the experiments download data and take hours, so they run for each release on GitHub Actions, which opens a pull request with the results, and anyone can run them again on their own machine.

## Papers

* `cheng_decision_2009`: W. Cheng, J. Hühn and E. Hüllermeier, "Decision tree and instance-based learning for label ranking", in Proceedings of the 26th International Conference on Machine Learning, Montreal, Canada, pp. 161-168, 2009.
  * Methods:
    * IBLR, `KNeighborsLabelRanker(model="mallows", weights="dudani")`.
    * LRT, `DecisionTreeLabelRanker()`, with a fixed `random_state` and `min_samples_split` twice the number of labels.
  * Datasets: authorship, bodyfat, calhousing, cpu (cpu-small in the paper), elevators, fried, glass, housing, iris, pendigits, segment, stock, vehicle, vowel, wine and wisconsin, with the `label ranking` tag, in Table 1 (p. 6), whose numbers of samples are about 1.8 times those of OpenML and of `cheng_label_2010`, as the report shows.
  * Scenarios: complete training rankings, and 30% and 60% of the labels of each training ranking removed at random.
  * Metric: Kendall's $\tau$.
  * Cross-validation: five repetitions of a ten-fold cross-validation (p. 7), and an inner five-fold cross-validation, since the paper does not specify it.
  * Preprocessing: none, since the features are already normalized in OpenML, as the paper requires (p. 6).
  * Search: for IBLR, a grid of the number of neighbors among 5, 10, 15 and 20, with the Euclidean distance; none for LRT, whose parameters the paper does not tune.
  * Published results: Table 2 (p. 8).
  * Not specified: the grid of numbers of neighbors, so that of `cheng_label_2010` is used, the number of folds of the inner cross-validation, and how the inner cross-validation scores the incomplete validation rankings, as described in the protocol below. The second rule that stops growing the tree, on nodes with fewer than $2|\mathcal{Y}| = 2n$ labels (p. 5), where $\mathcal{Y}$ is the set of the $n$ labels, is ambiguous: it is read as nodes with fewer than $2n$ samples, so `min_samples_split` is set to $2n$ in each training fold, rather than as nodes whose rankings $\sigma_i$ rank fewer than $2n$ labels in total, $\sum_i |\mathcal{Y}(\sigma_i)| < 2n$, which with complete rankings would only stop at single samples.
  * Notes: LRT differs from the paper by design in three points, so its scores may differ from the published ones even with the same data: each split of a node $t$ into children $t_L$ and $t_R$ minimizes the weighted mean of the impurities $I_{t_L}$ and $I_{t_R}$ of the children, with $I_t = 1/(1 + \hat\theta_t)$ for the estimated spread $\hat\theta_t$ of node $t$, instead of maximizing the weighted mean of their spreads $\hat\theta_{t_L}$ and $\hat\theta_{t_R}$ (equation (6), p. 5), which is infinite whenever a child has a single sample; the rule of $2|\mathcal{Y}|$ labels is read as described above; and, as in scikit-learn, a node becomes a leaf when its best split increases the impurity. For IBLR, the grid of numbers of neighbors most likely explains the lower scores in the scenarios with missing labels, since the more labels are removed, the more often the inner cross-validation chooses the largest number of neighbors of the grid. With missing labels, the published scores seem to need more neighbors than this grid allows, such as up to the square root of the number of samples, which brings the results closer to them on the largest datasets.
* `cheng_label_2010`: W. Cheng, K. Dembczyński and E. Hüllermeier, "Label ranking methods based on the Plackett-Luce model", in Proceedings of the 27th International Conference on Machine Learning, Haifa, Israel, pp. 215-222, 2010.
  * Methods:
    * IB-PL, `KNeighborsLabelRanker(model="plackett_luce")`.
    * IB-Mal, `KNeighborsLabelRanker(model="mallows")`, without distance weights, as the paper does for a fair comparison with IB-PL.
  * Datasets: authorship, bodyfat, calhousing, cpu (cpu-small in the paper), elevators, fried, glass, housing, iris, pendigits, segment, stock, vehicle, vowel, wine and wisconsin, with the `label ranking` tag, in Table 1 (p. 6).
  * Scenarios: complete training rankings, and 30% and 60% of the labels of each training ranking removed at random.
  * Metric: Kendall's $\tau$.
  * Cross-validation: five repetitions of a ten-fold cross-validation (p. 6), and an inner five-fold cross-validation, since the paper does not specify it.
  * Preprocessing: none, since the features are already normalized in OpenML, as the paper requires (p. 5).
  * Search: a grid of the number of neighbors among 5, 10, 15 and 20 (p. 5), with the Euclidean distance.
  * Published results: Table 2 (p. 7).
  * Not specified: the number of folds of the inner cross-validation, and how the inner cross-validation scores the incomplete validation rankings, as described in the protocol below.
  * Notes: IB-Mal on segment differs from the published score far more than the scores of the folds vary, while IB-PL in the same table and IBLR in `cheng_decision_2009`, with the same model, match theirs on the same data, so the difference does not seem to come from the method or the data.
* `alfaro_learning_2021`: J. C. Alfaro, J. A. Aledo and J. A. Gámez, "Learning decision trees for the partial label ranking problem", International Journal of Intelligent Systems, vol. 36, no. 2, pp. 890-918, 2021.
  * Methods:
    * IBPLR (KNN-E-D in the tables), `KNeighborsPartialLabelRanker(weights="distance")`, with the Euclidean distance and the neighbors weighted by the inverse of their distance (p. 905).
  * Datasets: authorship, blocks, breast, ecoli, glass, iris, letter, libras, pendigits, satimage, segment, vehicle, vowel, wine and yeast, with the `partial label ranking` tag, in Table 5 (p. 905). They match the table in their numbers of samples, features, labels and distinct bucket orders and in their mean number of buckets, except breast, with 106 samples in OpenML and 109 in the table, but the same 62 bucket orders and mean number of buckets.
  * Scenarios: complete training rankings, and 30% and 60% of the labels of each training ranking removed at random, as in `cheng_decision_2009` (p. 907).
  * Metric: the $\tau_x$ coefficient.
  * Cross-validation: five repetitions of a ten-fold cross-validation (p. 906), and an inner five-fold cross-validation (p. 905).
  * Preprocessing: none, since the paper does not mention any.
  * Search: the number of neighbors starts at five and is doubled while the mean inner $\tau_x$ improves, and a binary search then halves the range between the best number and the one that stopped the doubling, keeping the upper half if its midpoint improves the best score and the lower half otherwise (p. 905).
  * Published results: Tables 8, 9 and 10 (pp. 908-909).
  * Not specified: what the search does when the doubling exceeds the training samples of the inner folds, so that number of samples is tried instead, and how the inner cross-validation scores the incomplete validation rankings, as described in the protocol below.
  * Notes: the trees of the paper (PLRT) and its Zero-R baseline are not reproduced yet.

Each directory holds the protocol of its paper (`experiment.py`), which declares its datasets, scenarios, metric, cross-validation and methods, as the docstring of `workflow/experiments/__init__.py` describes, the published scores (`paper_results.csv`, with the metric of each one) and datasets (`paper_datasets.csv`), with the table and page they come from, and the results of each version of scikit-lr under `results/<version>/`. The workflow finds the papers by their `experiment.py`, so adding a paper only takes its directory. What the papers share is in `workflow/experiments/`: the datasets and the transformations of their rankings (`datasets.py`), the metrics (`metrics.py`) and the searches of the parameters (`search.py`), such as a grid or the doubling and binary search of `alfaro_learning_2021`, whose options each paper sets.

## Datasets

The datasets are downloaded from OpenML, where the authors of scikit-lr uploaded them with the `label ranking` or the `partial label ranking` tag, and they are named as there. `OPENML_DATASETS` in `workflow/experiments/datasets.py` lists their identifiers under each tag, since a name may have several versions in OpenML. Each paper uses some of the datasets of a tag, and `paper_datasets.csv` gives the name of each one in the paper.

## Protocol

Each job runs one repetition of the outer cross-validation of a paper. In each fold, the training rankings are transformed as the scenario says, for instance removing some of their labels, the parameters of the method that depend on the training data, if any, are set from them, the others are chosen by its search with the inner cross-validation of the paper, if it has one, and the method is scored with the metric of the paper on the test rankings. The same seeds are used for every method and scenario, so they all get the same splits and random transformations. `workflow/run_cell.py` implements it.

In the scenarios that remove labels, the inner cross-validation splits the incomplete training rankings of each outer fold, so its validation rankings are incomplete too, and the papers do not say how they score a prediction against them. The metrics of scikit-lr only compare complete rankings, and an incomplete ranking only says how it orders the labels that it ranks, so each prediction is scored with the metric of the paper on those labels alone, after making the positions of the predicted ranking dense among them. Each pair of labels that the validation ranking ranks then counts as with complete rankings, and the pairs with a label that it does not rank do not count. The validation rankings with fewer than two ranked labels give no pair to compare and are left out. The test rankings of the outer cross-validation are always complete, so the scores compared with the published ones only use complete rankings. `observed_score` in `workflow/experiments/metrics.py` implements it.

The papers do not specify every detail of their protocol, as listed above, so the results are expected to be close to the published ones, not equal.

## Results

Each job writes `results/<version>/<dataset>/<method>/<scenario>/rep<repetition>.json` with the score, the parameters set from the training data and those chosen by the inner cross-validation, the number of convergence warnings, and the wall and CPU times of each fold, the parameters of the ranker, its search with its options, the cross-validation, the versions of the main packages and the machine where it ran (processor, cores, memory, system, Python, threading libraries and their threads), since the jobs of a run may run on different machines. Snakemake measures the time, memory and input and output of each job in the `.benchmark.tsv` file next to it. Each run also records the exact packages of the environment that writes the reports (`environment.txt`).

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
