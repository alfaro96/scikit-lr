#!/bin/bash

set -e

# Show the changes made by the hooks that fix files, since the fixes
# are lost with the runner and have to be applied by hand
pre-commit run --all-files --show-diff-on-failure --color=always
