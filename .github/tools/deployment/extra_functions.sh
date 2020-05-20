# Define extra functions

function setup_test_venv
{
    # Create a new empty virtual environment to testing for macOS
    # platforms. On Linux the tests are run in a Docker container
    if [ $(uname) == "Darwin" ]; then
        $PYTHON_EXE -m venv test_venv
        source test_venv/bin/activate

        PYTHON_EXE=`which python`
        PIP_CMD="$PYTHON_EXE -mpip"
        python -m pip install --upgrade pip wheel
    fi
}

function teardown_test_venv
{
    if [ $(uname) == "Darwin" ]; then
        source venv/bin/activate
    fi
}
