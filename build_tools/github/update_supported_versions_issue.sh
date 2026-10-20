#!/bin/bash

set -e

# The issue of the workflow is the open one with its title that the workflow
# opened itself, so the issues that people open with the same label or a
# similar title are left alone. gh reads the repository and the token from
# GH_REPO and GH_TOKEN
title="Supported versions to update"
issue=$(gh issue list --state open --app github-actions --label "$LABEL" \
    --search "\"$title\" in:title" --json number,title \
    --jq ".[] | select(.title == \"$title\") | .number" | head -n 1)

date=$(date -u +%Y-%m-%d)

# An empty report may still come with the line break of its output
if [[ -n "${REPORT//[[:space:]]/}" ]]; then
    body="The workflow \"$WORKFLOW_NAME\" found these changes to the supported versions on $date, see [its run]($RUN_URL):

$REPORT

This issue was opened by the workflow, which updates it every week while there is something to change and closes it once there is nothing left."
    if [[ -n "$issue" ]]; then
        # Editing the issue notifies nobody, so a change to the list of what is
        # due, such as a new release of scikit-learn, is also commented
        previous=$(gh issue view "$issue" --json body --jq .body | grep "^\* " || true)
        current=$(echo "$REPORT" | grep "^\* " || true)
        gh issue edit "$issue" --body "$body"
        echo "Updated issue #$issue"
        if [[ "$current" != "$previous" ]]; then
            gh issue comment "$issue" --body "What is due changed on $date, see [its run]($RUN_URL):

$REPORT"
        fi
    else
        gh issue create --title "$title" --label "$LABEL" --body "$body"
    fi
elif [[ -n "$issue" ]]; then
    gh issue close "$issue" --comment "Nothing is left to change on $date, see [its run]($RUN_URL)."
fi
