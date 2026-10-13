#!/bin/bash

set -e

# Imported from the reproduction directory, so that the installed package is
# found instead of the sources of the repository
version=$(cd reproducibility && python -c "import sklr; print(sklr.__version__)")
branch="reproducibility-$version-$GITHUB_RUN_ID"

git switch --create "$branch"
git add reproducibility/*/results/"$version"
git -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit --quiet --message "docs: add the reproduction of the papers with scikit-lr $version"
git push "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" "$branch"

gh pr create --base "$BASE_BRANCH" --head "$branch" \
    --title "docs: add the reproduction of the papers with scikit-lr $version" \
    --body "Results of the reproduction of the papers with scikit-lr $version, from \
run $GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID of the \
Reproducibility workflow on $GITHUB_SHA. Review the reports in \
\`reproducibility/<paper>/results/$version/report.md\` before merging."
