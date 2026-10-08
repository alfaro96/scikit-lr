#!/bin/bash

set -e

python -c "import sklr; sklr.show_versions()"

# The documentation of a development version lists the changes merged since
# the last release, so their changelog fragments are rendered into the release
# history. The fragments of a release are rendered and removed before tagging
# it, so there are none left to render for the documentation of its tag
if python -c "import sys, sklr; sys.exit('.dev' not in sklr.__version__)"; then
    towncrier build --yes
fi

make -C doc html
