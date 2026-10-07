py.install_sources(
  [
    '__init__.py',
    'mallows.py',
  ],
  subdir: 'sklr/_distributions',
)

foreach name : ['_borda', '_extension', '_sorting']
  py.extension_module(
    name,
    name + '.pyx',
    cython_args: cython_args,
    install: true,
    subdir: 'sklr/_distributions',
  )
endforeach

py.extension_module(
  '_mallows',
  '_mallows.pyx',
  cython_args: cython_args,
  dependencies: [openmp_dep],
  install: true,
  subdir: 'sklr/_distributions',
)

subdir('tests')
