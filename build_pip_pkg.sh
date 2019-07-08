#!/usr/bin/env bash
# Code from Tensorflow Addons
# ==============================================================================
set -e
set -x

PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/__main__/"

function abspath() {
  cd "$(dirname $1)"
  echo "$PWD/$(basename $1)"
  cd "$OLDPWD"
}

function main() {
  DEST=${1}
  BUILD_FLAG=${2}

  if [[ -z ${DEST} ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  mkdir -p ${DEST}
  DEST=$(abspath "${DEST}")
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
  echo $(date) : "=== Using tmpdir: ${TMPDIR}"
  echo "=== Copy TensorFlow Scientific files"

  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}LICENSE "${TMPDIR}"
  rsync -avm -L --exclude='*_test.py' ${PIP_FILE_PREFIX}tensorflow_scientific "${TMPDIR}"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  if [[ -z ${BUILD_FLAG} ]]; then
    ${PYTHON_VERSION:=python} setup.py bdist_wheel > /dev/null
  else
    ${PYTHON_VERSION:=python} setup.py bdist_wheel "${2}" > /dev/null
  fi

  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
