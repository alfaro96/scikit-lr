# Makefile to simplify repetitive tasks

all: clean inplace test

# Clean up the temporary artifacts
# (excluding the editor's settings)
clean:
	git clean -xdf -e .devcontainer/ \
				   -e .vscode/ \
				   -e .vscode-server/ \
				   -e *.code-workspace

# Use the editable mode to avoid installing and compiling
# the complete project each time a source file is updated
install:
	pip install --verbose --no-build-isolation --editable .

# Create the shared object files (.so suffix) of
# the extension modules in the current directory
inplace:
	python setup.py build_ext --inplace
 
# Test the code, the interactive examples for the
# documentation and create a code coverage report

test-code: inplace
	pytest -l -v sklr --durations=$(or ${DURATIONS}, 20)

test-docs:
	pytest $(shell find docs -name "*.rst" | sort)

test-coverage:
	rm -rf coverage .coverage
	pytest -l -v --cov=sklr --cov-report=html:coverage

test: test-code test-docs

# Build the documentation
docs: inplace
	make -C docs html

# Build, pull and run the docker image

docker-build:
	make -C docker build

docker-pull:
	make -C docker pull

docker-bash:
	make -C docker bash

docker-notebook:
	make -C docker notebook

# Analyze the code of the main module and also the
# modified files with respect to the master branch

code-analysis:
	flake8 sklr

flake8-diff:
	git diff master -u -- "*.py" | flake8 --diff
