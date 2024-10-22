#!/bin/bash

set -e

echo Set up rpi 400 for FTC-24 Workshop


LIBIIO_REPO=https://github.com/analogdevicesinc/libiio
LIBM2K_REPO=https://github.com/analogdevicesinc/libm2k
PYADI_REPO=https://github.com/thorenscientific/pyadi-iio
GENALYZER_REPO=https://github.com/analogdevicesinc/genalyzer
WS_REPO=https://github.com/cristina-suteu/ftc24-ws

LIBIIO_VER=v0.26
LIBM2K_VER=7b31a3d
PYADI_VER=ad4080
GENALYZER_VER=7ab380d

STAGING_DIR=/home/analog/tmp
WORK_DIR=/home/analog/ftc24-ws

PIP=pip3


stg_dir() {
echo -- create staging directory
pushd /home/analog/
if [ -d "$STAGING_DIR" ]; then
	sudo rm -rf $STAGING_DIR
fi
mkdir $STAGING_DIR
	
}

pip() {
$PIP install numpy==1.20.4 obspy
}

libiio() {
echo -- installing libiio
pushd $STAGING_DIR
git clone $LIBIIO_REPO
cd libiio
git checkout $LIBIIO_VER
mkdir build && cd build
cmake -DWITH_SERIAL_BACKEND=ON -DPYTHON_BINDINGS=on ../
make 
sudo make install
popd
}

libm2k() {
echo -- installing libm2k
pushd $STAGING_DIR
git clone  $LIBM2K_REPO
cd libm2k
git checkout $LIBM2K_VER
mkdir build && cd build
cmake -DENABLE_PYTHON=on ../
make 
sudo make install 
popd
}

pyadi() {
echo -- installing pyadi-iio
pushd $STAGING_DIR
git clone $PYADI_REPO
cd pyadi-iio
git checkout $PYADI_VER
$PIP install .

}

genalyzer() {
echo -- installing genalyzer
pushd $STAGING_DIR
git clone $GENALYZER_REPO
cd genalyzer
git checkout $GENALYZER_VER
mkdir build && cd build 
cmake -DPYTHON_BINDINGS=on -DBUILD_TESTS_EXAMPLES=on ../
make 
sudo make install
popd
}
scopy2 () {
 echo -- installing Scopy 2.0
 pushd $STAGING_DIR
 wget https://github.com/analogdevicesinc/scopy/actions/runs/11399600398/artifacts/2072940114
 unzip scopy-linux-armhf-0d7e996.zip
 popd
}

setup_rpi() {
stg_dir
pip
libiio
libm2k
pyadi
genalyzer
#scopy2

}

setup_rpi
