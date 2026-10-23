#!/bin/bash

set -e

# The scheduled and manual runs always test. A pull request only tests when the
# message of its latest commit has the marker of the workflow, or when it changes
# one of the files of the workflow, given as a space separated list in PATHS,
# which must be tested before they reach master
if [[ "$EVENT_NAME" != "pull_request" ]]; then
    run=true
elif git log -1 --format=%B "$HEAD_SHA" | grep -qF "$MARKER"; then
    run=true
elif git diff --name-only "origin/$BASE_REF...$HEAD_SHA" \
    | grep -qxF -f <(tr ' ' '\n' <<< "$PATHS"); then
    run=true
else
    run=false
fi
echo "Run the tests: $run"
echo "run=$run" >> "$GITHUB_OUTPUT"
