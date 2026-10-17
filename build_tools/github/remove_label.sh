#!/bin/bash

set -e

gh pr edit "$PR_NUMBER" --repo "$GITHUB_REPOSITORY" --remove-label "$LABEL"
