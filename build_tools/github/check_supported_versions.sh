#!/bin/bash

set -e

python -m pip install packaging

report=$(python build_tools/github/check_supported_versions.py)
echo "${report:-Nothing to update}" | tee -a "$GITHUB_STEP_SUMMARY"

# The report spans several lines, so it is written as a multiline output, between
# two lines with a delimiter that the report cannot contain
delimiter="EOF-$(openssl rand -hex 16)"
{
    echo "report<<$delimiter"
    echo "$report"
    echo "$delimiter"
} >> "$GITHUB_OUTPUT"
