# Define custom utilities

# Stuff to do before starting building the wheels
function pre_build
{
    # No necessary, skip
    :
}

# Run tests on installed distribution from an empty directory
function run_tests
{
    pytest -l -v --pyargs sklr
}
