project(
  'scikit-lr',
  'c', 'cython',
  version: run_command('sklr/_build_utils/version.py', check: true).stdout().strip(),
  license: 'MIT',
  meson_version: '>= 1.9.0',
  default_options: [
    'c_std=c11',
  ],
)

py = import('python').find_installation(pure: false)

# pip only installs the build requirements of pyproject.toml when building in
# an isolated environment, so check their minimum versions for the builds
# without isolation, which use the packages already installed. scikit-learn is
# left out, since some builds use a release outside its range on purpose
min_version_script = 'sklr/_build_utils/min_version.py'

cython_version = meson.get_compiler('cython').version()
cython_min_version = run_command(py, [min_version_script, 'cython'], check: true).stdout().strip()
if not cython_version.version_compare('>=' + cython_min_version)
  error('scikit-lr requires Cython>=' + cython_min_version + ', got ' + cython_version + ' instead')
endif

numpy_version = run_command(py,
  ['-c', 'import importlib.metadata; print(importlib.metadata.version("numpy"))'],
  check: true
).stdout().strip()
numpy_min_version = run_command(py, [min_version_script, 'numpy'], check: true).stdout().strip()
if not numpy_version.version_compare('>=' + numpy_min_version)
  error('scikit-lr requires NumPy>=' + numpy_min_version + ', got ' + numpy_version + ' instead')
endif

# meson-python is not needed when running Meson directly,
# so it is only checked when it is installed
meson_python_version_result = run_command(py,
  ['-c', 'import importlib.metadata; print(importlib.metadata.version("meson-python"))'],
  check: false
)
if meson_python_version_result.returncode() == 0
  meson_python_version = meson_python_version_result.stdout().strip()
  meson_python_min_version = run_command(py, [min_version_script, 'meson-python'], check: true).stdout().strip()
  if not meson_python_version.version_compare('>=' + meson_python_min_version)
    error('scikit-lr requires meson-python>=' + meson_python_min_version + ', got ' + meson_python_version + ' instead')
  endif
endif

# Each meson.build installs the Python files of its directory, listed one by
# one, so the wheels only get the files that run (not the Cython sources, the
# build helpers or the bytecode left in the source tree)
subdir('sklr')
