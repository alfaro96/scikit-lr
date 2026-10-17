#!/bin/bash

set -e

remote="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

# The commit of the published site, empty before the first deployment
published=$(git ls-remote "$remote" refs/heads/gh-pages | cut -f1)

if [[ -n "$published" ]]; then
    git clone --quiet --depth 1 --branch gh-pages "$remote" "$SITE_DIR"
    rm -rf "$SITE_DIR/.git"
else
    mkdir "$SITE_DIR"
fi

python build_tools/github/update_doc_site.py

# The branch is replaced by a single commit with the whole site, so the clones
# of the repository do not download the pages of every past deployment
cd "$SITE_DIR"
git init --quiet --initial-branch gh-pages
git add --all
git -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit --quiet --message "Deploy the documentation of $GITHUB_REF_NAME ($GITHUB_SHA)"
git push --force-with-lease="gh-pages:$published" "$remote" gh-pages
