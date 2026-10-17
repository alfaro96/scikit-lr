# Build the documentation with Sphinx, where every warning is an error
# so that the pages that would render wrongly are not published
SPHINXOPTS ?= --fail-on-warning --keep-going
SPHINXBUILD ?= sphinx-build
BUILDDIR = _build

.PHONY: html clean

html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) . $(BUILDDIR)/html

# The pages generated from the docstrings and the examples are written
# outside of the build directory, so they are removed too
clean:
	rm -rf $(BUILDDIR) auto_examples modules/generated sg_execution_times.rst
