# Define extra functions

function setup_test_venv
{
    # Test in a new empty virtual environment
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
