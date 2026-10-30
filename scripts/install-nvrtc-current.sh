#!/usr/bin/env bash

# One immutable, MoM-owned compiler for all NVRTC users; never rewrite another SDK.
# The subshell confines temporary-file cleanup and shell options to this operation.
install_nvrtc_current() (
  set -euo pipefail
  umask 022
  [[ $(uname -m) == x86_64 ]] || { echo 'NVRTC requires Linux x86_64.' >&2; exit 1; }
  case ${2:-} in ''|--validate-only) ;; *) echo "Unknown NVRTC option: $2" >&2; exit 2;; esac
  nv_dest=${1:-/opt/mom/nvrtc-13.3.33}
  [[ $nv_dest == /* && $nv_dest != */ && $nv_dest != / &&
     ${nv_dest##*/} != . && ${nv_dest##*/} != .. ]] || {
    echo 'NVRTC destination must be an absolute directory path.' >&2; exit 2;
  }
  nv_version=13.3.33
  nv_prefix=cuda_nvrtc-linux-x86_64-$nv_version-archive
  nv_files=(lib/libnvrtc.so.$nv_version lib/libnvrtc-builtins.so.$nv_version LICENSE include/nvrtc.h)
  nv_sizes=(120072328 6555592 68070 56371)
  nv_hashes=(afff6fa2eae856b90f66c2d83bbce4a4177404c506291e3a83486244d86c2815
    7394c640e5761d13d2bbcdbc4b4c5dbac7cb53cd5bc732d78f8a5cb38638e913
    088381bc2d891e719a2a9398645b00bb45f3b24231473a8283ac7e3e66b8a028
    8d0373dc86a2c68528ddd13247204c82df670636383a0fe7f31b791195181f9d)
  nv_matches() {
    local actual
    [[ ! -L $1 && -f $1 && $(stat -Lc %s -- "$1") == "$2" ]] || return 1
    actual=$(sha256sum < "$1") || return 1
    [[ ${actual%% *} == "$3" ]]
  }
  nv_validate() {
    local i
    [[ ! -L $1 && -d $1 && ! -L $1/lib && ! -L $1/include ]] || return 1
    for i in 0 1 2 3; do
      nv_matches "$1/${nv_files[i]}" "${nv_sizes[i]}" "${nv_hashes[i]}" || return 1
    done
    [[ -L $1/lib/libnvrtc.so && $(readlink -- "$1/lib/libnvrtc.so") == libnvrtc.so.$nv_version &&
       -L $1/lib/libnvrtc.so.13 && $(readlink -- "$1/lib/libnvrtc.so.13") == libnvrtc.so.$nv_version &&
       -L $1/lib/libnvrtc-builtins.so.13.3 &&
       $(readlink -- "$1/lib/libnvrtc-builtins.so.13.3") == libnvrtc-builtins.so.$nv_version ]]
  }
  if [[ -e $nv_dest || -L $nv_dest ]]; then
    nv_validate "$nv_dest" || { echo "Refusing conflicting NVRTC directory: $nv_dest" >&2; exit 1; }
    exit 0
  fi
  [[ ${2:-} != --validate-only ]] || exit 1
  nv_parent=${nv_dest%/*}
  [[ -n $nv_parent ]] || nv_parent=/
  mkdir -p -- "$nv_parent" || exit 1
  nv_parent=$(CDPATH='' cd -- "$nv_parent" && pwd -P) || exit 1
  nv_dest=$nv_parent/${nv_dest##*/}
  nv_work=$(mktemp -d "$nv_parent/.mom-nvrtc-13.3.33.XXXXXX") || exit 1
  trap 'if [[ -n "$nv_work" && "$nv_work" == "$nv_parent"/.mom-nvrtc-13.3.33.* ]]; then rm -rf -- "$nv_work"; fi' EXIT
  curl -fsSL --proto '=https' --tlsv1.2 --retry 5 \
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-x86_64/$nv_prefix.tar.xz" \
    -o "$nv_work/archive.tar.xz" || exit 1
  nv_archive_hash=$(sha256sum < "$nv_work/archive.tar.xz") || exit 1
  [[ ${nv_archive_hash%% *} == 9e8f78278215babd1236b137252424ca7912c185bd093201f5d97f7dd763b74a ]] || {
    echo 'NVRTC archive SHA256 mismatch' >&2; exit 1;
  }
  nv_members=()
  for nv_file in "${nv_files[@]}"; do nv_members+=("$nv_prefix/$nv_file"); done
  tar --no-same-owner --no-same-permissions -xJf "$nv_work/archive.tar.xz" \
    -C "$nv_work" "${nv_members[@]}" || exit 1
  nv_stage=$nv_work/$nv_prefix
  for nv_file in "${nv_files[@]}"; do chmod 644 -- "$nv_stage/$nv_file" || exit 1; done
  chmod 755 -- "$nv_stage" "$nv_stage/lib" "$nv_stage/include" || exit 1
  ln -s -- "libnvrtc.so.$nv_version" "$nv_stage/lib/libnvrtc.so" || exit 1
  ln -s -- "libnvrtc.so.$nv_version" "$nv_stage/lib/libnvrtc.so.13" || exit 1
  ln -s -- "libnvrtc-builtins.so.$nv_version" "$nv_stage/lib/libnvrtc-builtins.so.13.3" || exit 1
  nv_validate "$nv_stage" || { echo 'NVRTC payload validation failed' >&2; exit 1; }
  # Publish the complete tree atomically, without overwriting a concurrent installer.
  # A skipped rename can return either status across mv versions. Only the
  # complete, independently verified destination below can establish success.
  mv -T --no-clobber -- "$nv_stage" "$nv_dest" || [[ -d $nv_dest && ! -L $nv_dest ]] || exit 1
  nv_validate "$nv_dest" || { echo 'Published NVRTC directory is invalid' >&2; exit 1; }
  echo "NVRTC $nv_version is ready at $nv_dest; other CUDA SDKs are unchanged."
)
