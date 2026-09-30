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

# Install the sklr directory as is, so new Python files are picked up
# without listing each of them (as py.install_sources() would require)
install_subdir('sklr', install_dir: py.get_install_dir())

subdir('sklr')
