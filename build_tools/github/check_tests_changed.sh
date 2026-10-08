#!/bin/bash

set -e

# The changelog entry is only required when the pull request changes a test
# file, compared with its base branch
changed_files=$(git diff --name-only "origin/$BASE_REF")
if grep -qE '^sklr/.+/test_[^/]+\.py$' <<< "$changed_files"; then
    echo "check_changelog=true" >> "$GITHUB_OUTPUT"
fi
