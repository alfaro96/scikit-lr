#!/bin/bash

# This script is meant to be called by the "Execute tests" step defined in
# linting.yml. The behaviour of the script is controlled by the named step
# defined in the linting.yml in the folder .github/workflows of the project.

# Exit immediately if a command
# exits with a non-zero status
set -e

# Define the name of the repository
# and the URL of the project
PROJECT=alfaro96/scikit-lr
PROJECT_URL=https://github.com/$PROJECT.git

# Find the remote with the project name (origin in most cases)
REMOTE=$(git remote -v | grep $PROJECT | cut -f1 | head -1 || echo "")
echo "Remotes"
git remote --verbose

# Obtain the reference to the latest commits
# of $LOCAL_BRANCH_REF and $REMOTE/master
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
# and, if it does not exist, show the corresponding message to user
COMMIT=$(git merge-base $LOCAL_BRANCH_REF $REMOTE_MASTER_REF) || \
    echo "No common ancestor found."

# Exit as far as no common ancestor is found
if [ -z "$COMMIT" ]; then
    exit 1
fi

# Obtain the short hash of the common commit ancestor
COMMIT_SHORT_HASH=$(git rev-parse --short $COMMIT)
echo "Common ancestor is $COMMIT_SHORT_HASH."
git --no-pager show --no-patch $COMMIT_SHORT_HASH

# Find the commit range between the
# common ancestor and the local commit
COMMIT_RANGE="$COMMIT_SHORT_HASH..$LOCAL_BRANCH_SHORT_HASH"

# Find the files that have been modified in the
# local commit with respect to the common ancestor
MODIFIED_FILES="$(git diff --name-only $COMMIT_RANGE || \
    echo "no_match")"

# Show the proper message to the user
# when there are no files being modified
if [[ "$MODIFIED_FILES" == "no_match" ]]; then
    echo "No files has been modified."
else
    git diff --unified=0 $COMMIT_RANGE -- $MODIFIED_FILES | \
    flake8 --diff --show-source "$*"
fi

# Inform to the user if no
# problems have been detected
echo "No problem detected by flake8."
