function setup_compiler {
    # Install OpenMP support on macOS
    if [ $(uname) == "Darwin" ]; then
        brew install libomp
        export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
        export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
        export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
        export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
        export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib
    fi
}

function teardown_compiler {
    if [ $(uname) == "Darwin" ]; then
        brew uninstall libomp
    fi
}

function setup_test_venv {
    # Create a new empty venv dedicated to testing for non-Linux platforms. On
    # Linux the tests are run in a Docker container.
    if [ $(uname) != "Linux" ]; then
        deactivate || echo ""
        $PYTHON_EXE -m venv test_venv
        if [ $(uname) == "Darwin" ]; then
            source test_venv/bin/activate
        else
            source test_venv/Scripts/activate
        fi
        # Defined python to be within this env
        PYTHON_EXE=`which python`
        PIP_CMD="$PYTHON_EXE -mpip"
        # Note: the idiom "python -m pip install ..." is necessary to upgrade
        # pip itself on Windows. Otherwise one would get a permission error on
        # pip.exe.
        python -m pip install --upgrade pip wheel
        if [ "$TEST_DEPENDS" != "" ]; then
            pip install $TEST_DEPENDS
        fi
    fi
}

function teardown_test_venv {
    if [ $(uname) != "Linux" ]; then
        deactivate || echo ""
        if [ $(uname) == "Darwin" ]; then
            source venv/bin/activate
        fi
    fi
}
