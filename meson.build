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

# Each meson.build installs the Python files of its directory, listed one by
# one, so the wheels only get the files that run (not the Cython sources, the
# build helpers or the bytecode left in the source tree)
subdir('sklr')
