# Makefile to simplify repetitive tasks

# All command clean useless files and directories, install
# the package (in place) and run the tests (code and coverage)
all: clean inplace test

# Remove useless files and directories
clean:
	python setup.py clean

# Installers

# Build extensions in place (folder of the source code)
inplace:
	python setup.py build_ext -i

# Locally install the package
install:
	python setup.py install --prefix=$(HOME)/.local

# Build src files (Cython)
cython:
	python setup.py build_src

# Testing

# Test the code
test-code: inplace
	pytest -l -v sklr --durations=20

# Test the coverage
test-coverage:
	rm -rf coverage .coverage
	pytest -l -v --cov=sklr --cov-report=html:coverage

# Test the code and the coverage
test: test-code test-coverage
