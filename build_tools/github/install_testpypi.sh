#!/bin/bash

set -e

python -m pip install pytest

# A new version may take a few minutes to be listed in the index of TestPyPI
# after its upload, so the installation is retried for a while. The index
# allows caching its pages for some minutes, so pip must not cache them, or
# the retries would read the page without the version
for attempt in {1..10}; do
    # Install only scikit-lr from TestPyPI, without its dependencies, because
    # anyone can upload a package with the name of a dependency there. The
    # source distribution is refused, so a missing wheel makes it fail
    # instead of compiling the package and testing something else
    if python -m pip install --verbose --no-cache-dir --no-deps \
        --only-binary :all: --index-url https://test.pypi.org/simple/ \
        "scikit-lr==$VERSION"; then
        break
    fi
    if [[ $attempt -eq 10 ]]; then
        exit 1
    fi
    sleep 60
done

# Install the dependencies from PyPI: scikit-lr is already installed,
# so pip only installs what it requires
python -m pip install "scikit-lr==$VERSION"
