# Makefile to simplify repetitive tasks

# All clean the files, install the package
# (in place) and run the tests (code and coverage)
all: clean inplace test

# Clean
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

# Code
test-code: inplace
	pytest -l -v plr --durations=20

# Coverage
test-coverage:
	rm -rf coverage .coverage
	pytest -l -v --cov=plr --cov-report=html:coverage

# All
test: test-code test-coverage
