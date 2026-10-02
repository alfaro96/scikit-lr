# scikit-learn has no get_include(), so pass the directory that contains
# its installed package to find the .pxd files it ships. find_spec does
# not import scikit-learn, it only locates the package
sklearn_include_dir = run_command(py,
  [
    '-c',
    '''
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("sklearn")
print(Path(spec.submodule_search_locations[0]).parent)
'''
  ],
  check: true
).stdout().strip()

cython_args = ['--include-dir', sklearn_include_dir]

if get_option('linetrace')
  # Cython only emits the calls that report each executed line when the
  # directive is set, and the C code only compiles them in with the macro,
  # which also covers the code that runs without the GIL. The plugin that
  # coverage.py needs for Cython only works with its tracer based on
  # sys.settrace, which does not receive the sys.monitoring events that
  # Cython reports by default on the newer Python versions
  cython_args += ['-X', 'linetrace=True']
  add_project_arguments(
    '-DCYTHON_TRACE_NOGIL=1',
    '-DCYTHON_USE_SYS_MONITORING=0',
    language: 'c',
  )
endif

# Checked against the scikit-learn found at runtime when importing sklr
sklearn_build_version = run_command(py,
  ['-c', 'import importlib.metadata; print(importlib.metadata.version("scikit-learn"))'],
  check: true
).stdout().strip()

configure_file(
  input: '_build_utils/_build_info.py.in',
  output: '_build_info.py',
  configuration: {'SKLEARN_BUILD_VERSION': sklearn_build_version},
  install: true,
  install_dir: py.get_install_dir() / 'sklr',
  # Meson does not tag the generated files, which would otherwise
  # be left out when installing only the runtime files
  install_tag: 'python-runtime',
)

py.install_sources(
  [
    '__init__.py',
    '_sklearn_version.py',
    'conftest.py',
  ],
  subdir: 'sklr',
)

subdir('tests')
subdir('__check_build')
subdir('utils')
