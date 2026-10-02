#!/bin/bash

cat << EOF
A pull request that affects users needs a changelog entry that describes its
changes. Otherwise, there is nothing to do: a maintainer will add the "no
changelog needed" label, which makes this check pass.

See how to write a changelog entry in the contributing guide:
https://alfaro96.github.io/scikit-lr/dev/developers/contributing.html#changelog
EOF

# The step only runs when the check of the entry failed, which must still fail the job
exit 1
