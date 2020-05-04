# Makefile to simplify repetitive tasks

all: clean inplace test

# Clean up temporary files
clean:
	python setup.py clean
	rm -rf dist

# Build the extensions "in-place"
inplace:
	python setup.py build_ext -i

# Test the code and get the slowest durations
test-code: inplace
	pytest -l -v sklr --durations=$(or ${DURATIONS}, 20)

# Produce a coverage report in HTML format
test-coverage:
	rm -rf coverage .coverage
	pytest -l -v --cov=sklr --cov-report=html:coverage

# Test the interactive examples in the documentation
test-docs:
	pytest $(shell find docs -name "*.rst" | sort)

test: test-code test-docs

# Create the documentation in HTML format
docs: inplace
	make -C docs html

# Analyze the style of the code
code-analysis:
	flake8 sklr | grep -v __init__ || true

# Find the changed files and analyze their style
flake8-diff:
	git diff master -u -- "*.py" | flake8 --diff
