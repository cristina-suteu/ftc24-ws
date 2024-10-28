#!/bin/bash

set -e

echo Set up rpi 400 for FTC-24 Workshop


LIBIIO_REPO=https://github.com/analogdevicesinc/libiio
LIBM2K_REPO=https://github.com/analogdevicesinc/libm2k
PYADI_REPO=https://github.com/cristina-suteu/pyadi-iio
GENALYZER_REPO=https://github.com/analogdevicesinc/genalyzer
WS_REPO=https://github.com/cristina-suteu/ftc24-ws

LIBIIO_VER=v0.26
LIBM2K_VER=7b31a3d
PYADI_VER=ad4080
GENALYZER_VER=7ab380d

STAGING_DIR=/home/analog/tmp
WORK_DIR=/home/analog/tmp/ftc24-ws

PIP=pip3

set_date() {
echo -- setting date and time
sudo date -s "$(wget --method=HEAD -qSO- --max-redirect=0 google.com 2>&1 | sed -n 's/^ *Date: *//p')"
}

stg_dir() {
echo -- create staging directory
pushd /home/analog/
if [ -d "$STAGING_DIR" ]; then
	sudo rm -rf $STAGING_DIR
fi
mkdir $STAGING_DIR
}

work_dir() {
echo -- create working directory
pushd $STAGING_DIR
git clone $WS_REPO
cp -r $WORK_DIR /home/analog/Desktop
popd
}

pip() {
sudo $PIP uninstall numpy matplotlib
$PIP install matplotlib==3.9.2 obspy 
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
wget  --header='User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0' --header='Accept-Language: en-US,en;q=0.5' --header='Connection: keep-alive' --header='Cache-Control: no-cache' https://swdownloads.analog.com/cse/workshops/ftc2024/scopy-linux-armhf-43fef19.zip
unzip scopy2-prerelease-8c70563-linux-armhf.zip
cd $STAGING_DIR/scopy-linux-armhf-8c70563
sudo chmod +x Scopy-armhf.AppImage
cp Scopy-armhf.AppImage /home/analog/Desktop
popd
}

setup_rpi() {
set_date
stg_dir
work_dir
pip
libiio
libm2k
pyadi
genalyzer
scopy2
sudo ldconfig

}

setup_rpi


