# Makefile to simplify repetitive tasks

# Environment variables
PYTHON  ?= python
DOCKER  ?= docker
PYTEST  ?= pytest
ROOTDIR ?= ${PWD}

# All, clean the files, install (in place) the package and run the tests
all: clean inplace test

# Clean
clean:
	$(PYTHON) setup.py clean

# Docker

# Build the image
docker-build:
	$(DOCKER) build -t alfaro96/plr:development .

# Run the image
docker-run:
	$(DOCKER) run -ti -v $(ROOTDIR)/:/home/plr/workspace/ --rm alfaro96/plr:development

# Both
docker: docker-build docker-run

# Installers

# Build extensions in place (folder of the source code)
inplace:
	$(PYTHON) setup.py build_ext -i

# Build src files (Cython)
cython:
	$(PYTHON) setup.py build_src

# Testing

# Code
test-code:
	$(PYTEST) -l -v plr

# Coverage
test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) -l -v --cov=plr --cov-report=html:coverage

# All
test: test-code test-coverage
