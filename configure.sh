#!/usr/bin/env bash
# Code from Tensorflow Addons
# ==============================================================================
# Usage: configure.sh [--quiet]
#
# Options:
#  --quiet  Give less output.

QUIET_FLAG=""
if [[ $1 == "--quiet" ]]; then
    QUIET_FLAG="--quiet"
elif [[ ! -z "$1" ]]; then
    echo "Found unsupported args: $@"
    exit 1
fi

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

[[ -f .bazelrc ]] && rm .bazelrc
read -r -p "Tensorflow will be upgraded to 2.0. Are You Sure? [Y/n] " reply
case $reply in
    [yY]*) echo "Installing...";;
    * ) echo "Goodbye!"; exit;;
esac
${PYTHON_VERSION:=python} -m pip install $QUIET_FLAG -r requirements.txt

TF_CFLAGS=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"
TF_CXX11_ABI_FLAG=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)') )

SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_CXX11_ABI_FLAG}
