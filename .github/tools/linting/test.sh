#!/bin/bash

PROJECT=alfaro96/scikit-lr
PROJECT_URL=https://github.com/$PROJECT.git

# Immediately exit with a non-zero status command
set -e

# Find the remote with the project name (upstream in most cases)
REMOTE=$(git remote -v | grep $PROJECT | cut -f1 | head -1 || echo "")
echo "Remotes"
git remote --verbose

# Get the local branch reference and the remote upstream reference
LOCAL_BRANCH_REF=$(git rev-parse --abbrev-ref HEAD)
REMOTE_MASTER_REF="$REMOTE/master"
echo "Last two commits in $LOCAL_BRANCH_REF"
git --no-pager log -2 $LOCAL_BRANCH_REF

# Make sure that $REMOTE_MASTER_REF is a valid reference
echo "Fetching $REMOTE_MASTER_REF"
git fetch $REMOTE master:refs/remotes/$REMOTE_MASTER_REF

# Find the short hash of the latest commits
LOCAL_BRANCH_SHORT_HASH=$(git rev-parse --short $LOCAL_BRANCH_REF)
REMOTE_MASTER_SHORT_HASH=$(git rev-parse --short $REMOTE_MASTER_REF)

# Find the common ancestor between $LOCAL_BRANCH_REF and $REMOTE/master
COMMIT=$(git merge-base $LOCAL_BRANCH_REF $REMOTE_MASTER_REF) || \
    echo "No common ancestor found."

if [ -z "$COMMIT" ]; then
    exit 1
fi

# Obtain the short hash of the common commit ancestor
COMMIT_SHORT_HASH=$(git rev-parse --short $COMMIT)
echo "Common ancestor is $COMMIT_SHORT_HASH."
git --no-pager show --no-patch $COMMIT_SHORT_HASH

# Find the commit range between the common ancestor and the local commit
COMMIT_RANGE="$COMMIT_SHORT_HASH..$LOCAL_BRANCH_SHORT_HASH"

# Find the modified files in the local commit with respect to
# the common ancestor and exit in case that there is no match
MODIFIED_FILES="$(git diff --name-only $COMMIT_RANGE || \
    echo "no_match")"

if [[ "$MODIFIED_FILES" == "no_match" ]]; then
    echo "No files has been modified."
else
    git diff --unified=0 $COMMIT_RANGE -- $MODIFIED_FILES | \
    flake8 --diff --show-source "$*"
fi

echo "No problem detected by flake8."
