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

subdir('__check_build')
