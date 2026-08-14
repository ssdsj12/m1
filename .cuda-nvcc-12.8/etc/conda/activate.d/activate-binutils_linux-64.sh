# shellcheck shell=sh

# This function takes no arguments
# It tries to determine the name of this file in a programatic way.
_get_sourced_filename() {
    # shellcheck disable=SC3054,SC2296 # non-POSIX array access and bad '(' are guarded
    if [ -n "${BASH_SOURCE+x}" ] && [ -n "${BASH_SOURCE[0]}" ]; then
        # shellcheck disable=SC3054 # non-POSIX array access is guarded
        basename "${BASH_SOURCE[0]}"
    elif [ -n "$ZSH_NAME" ] && [ -n "${(%):-%x}" ]; then
        # in zsh use prompt-style expansion to introspect the same information
        # see http://stackoverflow.com/questions/9901210/bash-source0-equivalent-in-zsh
        # shellcheck disable=SC2296  # bad '(' is guarded
        basename "${(%):-%x}"
    else
        echo "UNKNOWN FILE"
    fi
}

# The format for args are name,value. name is the name of
#  the environment variable. The original value is stored
#  in environment variable CONDA_BACKUP_NAME
_tc_activation() {
  local thing
  local newval
  local from
  local to

  from=""
  to="CONDA_BACKUP_"

  for thing in "$@"; do
    case "${thing}" in
      *,*)
        newval="${thing#*,}"
        thing="${thing%%,*}"
        ;;
      *)
        echo "ERROR: unrecognized argument to activation function"
        return 1
        ;;
    esac
    eval oldval="\${$thing:-}"
    if [ -n "${oldval:-}" ]; then
      eval export "${to}'${thing}'=\"${oldval}\""
    else
      eval unset '${to}${thing}'
    fi
    eval export "'${from}${thing}=${newval}'"
  done
  return 0
}

if [ "0" = "1" ]; then
  CONDA_PREFIX=$(echo "${CONDA_PREFIX:-}" | sed 's,\\,\/,g')
fi

if [ "${CONDA_BUILD:-0}" = "1" ]; then
  if [ -f /tmp/old-env-$$.txt ]; then
    rm -f /tmp/old-env-$$.txt || true
  fi
  env > /tmp/old-env-$$.txt
fi

_tc_activation  \
  "ADDR2LINE,x86_64-conda-linux-gnu-addr2line" \
  "AR,x86_64-conda-linux-gnu-ar" \
  "CXXFILT,x86_64-conda-linux-gnu-c++filt" \
  "ELFEDIT,x86_64-conda-linux-gnu-elfedit" \
  "NM,x86_64-conda-linux-gnu-nm" \
  "OBJCOPY,x86_64-conda-linux-gnu-objcopy" \
  "OBJDUMP,x86_64-conda-linux-gnu-objdump" \
  "RANLIB,x86_64-conda-linux-gnu-ranlib" \
  "READELF,x86_64-conda-linux-gnu-readelf" \
  "SIZE,x86_64-conda-linux-gnu-size" \
  "STRINGS,x86_64-conda-linux-gnu-strings" \
  "STRIP,x86_64-conda-linux-gnu-strip" \
  "AS,x86_64-conda-linux-gnu-as" \
  "GPROF,x86_64-conda-linux-gnu-gprof" \
  "LD,x86_64-conda-linux-gnu-ld"

if [ $? -ne 0 ]; then
  echo "ERROR: $(_get_sourced_filename) failed, see above for details"
#exit 1
else
  if [ "${CONDA_BUILD:-0}" = "1" ]; then
    if [ -f /tmp/new-env-$$.txt ]; then
      rm -f /tmp/new-env-$$.txt || true
    fi
    env > /tmp/new-env-$$.txt

    echo "INFO: $(_get_sourced_filename) made the following environmental changes:"
    diff -U 0 -rN /tmp/old-env-$$.txt /tmp/new-env-$$.txt | tail -n +4 | grep "^-.*\|^+.*" | grep -v "CONDA_BACKUP_" | sort
    rm -f /tmp/old-env-$$.txt /tmp/new-env-$$.txt || true
  fi
fi

if [ "0" = "1" ]; then
  CONDA_PREFIX=$(echo "${CONDA_PREFIX:-}" | sed 's,\/,\\,g')
fi
