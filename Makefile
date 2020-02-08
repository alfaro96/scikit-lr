# Makefile to simplify repetitive tasks

# All command clean useless files and directories, install
# the package (in place) and run the tests (code and coverage)
all: clean inplace test

# Remove useless files and directories
clean:
	python setup.py clean

# Installers

# Build extensions in place
# (folder of the source code)
inplace:
	python setup.py build_ext -i

# Install the package locally
install:
	python setup.py install --prefix=$(HOME)/.local

# Build the source Cython files
cython:
	python setup.py build_src

# Testing

# Test the code
test-code: inplace
	pytest -l -v sklr --durations=20

# Test the coverage and create a beautiful report in HTML
test-coverage:
	rm -rf coverage .coverage
	pytest -l -v --cov=sklr --cov-report=html:coverage

# Test the code and the coverage
test: test-code test-coverage
