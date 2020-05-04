# Makefile for Sphinx documentation

# These variables can be set from the command line
# and also from the environment, for the first two
SPHINXOPTS ?= -j auto
SPHINXBUILD ?= sphinx-build
SOURCEDIR = source
BUILDDIR = _build

# Put this first so that "make" without argument is like "make help"
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: Route all the unknown targets to Sphinx using the
# new "make mode" option. $(O) is meant as shortcut for $(SPHINXOPTS)
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
