#!/bin/bash

set -e

# The issue of the workflow is the open one with its title that the workflow
# opened itself, so the issues that people open with the same label or a
# similar title are left alone. gh reads the repository and the token from
# GH_REPO and GH_TOKEN
title="CI failed on $WORKFLOW_NAME"
issue=$(gh issue list --state open --app github-actions --label "$LABEL" \
    --search "\"$title\" in:title" --json number,title \
    --jq ".[] | select(.title == \"$title\") | .number" | head -n 1)

date=$(date -u +%Y-%m-%d)

# A cancelled run says nothing about the code, so only a failure or a success
# changes the issue
if [[ "$RESULT" == "failure" ]]; then
    body="The workflow \"$WORKFLOW_NAME\" failed on $date, see [its run]($RUN_URL).

This issue was opened by the workflow, which updates it while it keeps failing and closes it once it passes again."
    if [[ -n "$issue" ]]; then
        gh issue edit "$issue" --body "$body"
        echo "Updated issue #$issue"
    else
        gh issue create --title "$title" --label "$LABEL" --body "$body"
    fi
elif [[ "$RESULT" == "success" && -n "$issue" ]]; then
    gh issue close "$issue" --comment "The workflow passed again on $date, see [its run]($RUN_URL)."
fi
