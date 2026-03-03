@ECHO OFF
pushd %~dp0
sphinx-build -M html . _build
popd
