#! /bin/bash

topdir=$(pwd)

git submodule update --init

libtoolize

cd $topdir/modules/hwloc
libtoolize
cd $topdir/modules/json-c
libtoolize
cd $topdir/modules/libfabric
libtoolize
cd $topdir/modules/ucx
libtoolize
cd $topdir/modules/yaksa
libtoolize

cd $topdir

./autogen.sh

set FC=$F90
set FCFLAGS=$F90FLAGS
unset F90
unset F90FLAGS

./configure --prefix=/ascldap/users/dkruse/local \
    --enable-fortran=no \
    --enable-g=none \
    --enable-error-checking=no \
    --enable-thread-cs=per-vci \
    --with-ch4-max-vcis=60 \
    --with-device=ch4:ucx \
    --enable-fast=O3 \
    --with-thread-package=argobots \
        CFLAGS="-I/ascldap/users/dkruse/local/include" \
        LDFLAGS="-L/ascldap/users/dkruse/local/lib" \
    2>&1 | tee configure.out
