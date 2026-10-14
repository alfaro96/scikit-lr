py.install_sources(
  [
    '__init__.py',
    'bucket_pivot.py',
  ],
  subdir: 'sklr/_aggregation',
)

py.extension_module(
  '_bucket_pivot',
  '_bucket_pivot.pyx',
  cython_args: cython_args,
  dependencies: [openmp_dep],
  install: true,
  subdir: 'sklr/_aggregation',
)

subdir('tests')
