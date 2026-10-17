py.install_sources(
  [
    '__init__.py',
    '_classes.py',
  ],
  subdir: 'sklr/tree',
)

py.extension_module(
  '_criterion',
  '_criterion.pyx',
  cython_args: cython_args,
  install: true,
  subdir: 'sklr/tree',
)

subdir('tests')
