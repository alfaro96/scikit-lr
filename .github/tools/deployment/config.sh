# Define custom utilities

function pre_build
{
    # Stuff to do before starting building the wheels
    :
}

function run_tests
{
    # Run tests on installed distribution from an empty directory
    pytest -l -v --pyargs sklr
}
