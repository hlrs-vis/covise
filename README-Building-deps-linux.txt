list of dependencies to get a appropriate/base linux build


Debian GNU/Linux buster/testing (16/04/2019)
--------------------------------------------

sources.list:

deb http://deb.debian.org/debian/ buster main non-free
deb-src http://deb.debian.org/debian/ buster main
deb http://security.debian.org/debian-security buster/updates main non-free
deb-src http://security.debian.org/debian-security buster/updates main

qtbase5-dev
libqt5
qttools5-dev-tools
qttools5-dev
openssl
libglew-dev
zlib1g-dev
libjpeg-dev
libpng-dev
libpng++-dev
libtiff5-dev
freeglut3-dev
libboost-dev
libboost-dev linphone-dev
libboost-dev
liblinphone-dev
libxerces-c-dev
libxmltooling-dev
libqt5quick5
libtbb-dev
libturbojpeg0-dev
libqt5quicktest5
libqt5quickparticles5
libboost-thread-dev
libboost-system-dev
libboost-iostreams-dev
libboost-filesystem-dev
qtdeclarative5-dev
libosgearth-dev
libopenthreads-dev
libopenscenegraph-3.4-dev
osgearth
osgearth-data
libboost-all-dev
libgdcm2-dev
libgdcm-tools
libmediastreamer-dev
libqt4-dev-bin
pyqt5-dev-tools
inventor-dev
libinventor1 
inventor-clients 
inventor-data 
inventor-demo 
inventor-dev 
inventor-doc 
libxerces-c-dev 
libxerces-c-doc 
libxerces-c-samples 
libtiff-dev 
libglew-dev 
libglew2.0
libglew-dev
libblas-dev  
liblapack-dev
tk8.6-dev
tk-dev
libavutil-dev
libavformat-dev
libswscale-dev
libgdal-dev
libcfitsio-dev
libvolpack1-dev
libzip-dev
libbison-dev
libopencv-dev
libbullet-dev
libusb-1.0-0-dev
libsoqt520-dev

Ubuntu 18.04 LTS Bionic Beaver
------------------------------

build-essential
git
gfortran
libopenmpi-dev
openmpi-common
openmpi-bin
cmake
libblas-dev
liblapack-dev
petsc-dev
openfoam
libQt5WebKitWidgets5
libQt5Gui5
libQt5Svg5
libQt5PrintSupport5
libQt5UiTools5
libQt5Script5
libQt5ScriptTools5
libqt5webkitwidgets5
libqt5gui5
libqt5svg5
libqt5printsupport5
libqt5uitools5
libqt5script5
libqt5scripttools5
libqt5core5a
libqt5network5
libqt5xml5
libqt5widgets5
libqt5webkit5
libqt5gui5
libqt5svg5
libqt5printsupport5
libqt5script5
libqt5scripttools5
libboost-all-dev
python3
libopenscenegraph-3.4-131
libopenscenegraph-3.4-dev
openscenegraph-3.4
openscenegraph-3.4-doc
openscenegraph-3.4-examples
libinventor1
inventor-clients
inventor-data
inventor-demo
inventor-dev
inventor-doc
libxerces-c-dev
libxerces-c-doc
libxerces-c-samples
libtiff-dev 
freeglut3
freeglut3-dev
qttools5-dev
qtscript5-dev
libqt5scripttools5
libqt5svg5-dev
libqt5opengl5-dev
libqt5webkit5-dev
libglew-dev
libglew2.0
pyqt5-dev
pyqt5-dev-tools




gdal:
./configure --prefix=/data/extern_libs/rhel8/gdal --with-cpp14 --with-poppler '--with-lzma' '--with-kml' 

xenomai:
git clone https://source.denx.de/Xenomai/xenomai.git
scripts/bootstrap
./configure --prefix=$EXTERNLIBS/xenomai -enable-dlopen-libs -enable-smp -with-core=mercury -enable-debug=symbols -enable-pshared --disable-testsuite --disable-demo

spack:
checkout to /sw/.../vis/spack
git clone https://github.com/spack/spack.git
git clone https://github.com/hlrs-vis/spack-hlrs-vis.git
export http_proxy=socks5h://localhost:1082
export HTTPS_PROXY=socks5h://localhost:1082

. spack/share/spack/setup-env.sh
spack repo add spack-hlrs-vis
spack external find

adjust .spack/packages.yaml
qt:
    buildable: false
    externals:
    - spec: qt@5.15.0+opengl
      prefix: /usr
    - spec: opengl@4.6.0
      prefix: /usr


adjust .spack/config.yaml:

config:
          # This is the path to the root of the Spack install tree.
          #   # You can use $spack here to refer to the root of the spack instance.
    install_tree:
        root: /sw/vulcan-CentOS8/hlrs/non-spack/vis/spack
    build_stage:
    - ~/ws/spack/stage
    build_jobs: 64

spack install -v covise target=nehalem

to run:
qsub -l select=1:node_type=vis,walltime=00:30:00 -X -I -q smp
module load vis/VirtualGL/2.6.5
turbovncserver
. /sw/vulcan-CentOS8/hlrs/non-spack/vis/spack/share/spack/setup-env.sh

#proj
cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/proj

#OpenCascade
mkdir build; cd build
dnf install tcl-devel tk-devel
cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/OpenCascade

#OpenSceneGraph
cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/openscenegraph -DCMAKE_PREFIX_PATH=${EXTERNLIBS}/OpenCascade
cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/openscenegraph -DCMAKE_PREFIX_PATH=${EXTERNLIBS}/fbx

#cla3d (from hlrs-vis)
./configure --prefix=/data/extern_libs/rhel9/cal3d
#libcitygml
cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/libcitygml -DCMAKE_PREFIX_PATH=${EXTERNLIBS}/openscenegraph

#osgCal (from hlrs-vis)
cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/osgcal -DCMAKE_PREFIX_PATH=${EXTERNLIBS}/cal3d
cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/osgcal -DCMAKE_PREFIX_PATH=${EXTERNLIBS}/openscenegraph

#osgEphemeris (from hlrs-vis)
cmake ../osgEphemeris -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/osgEphemeris -DOSGInstallLocation=${EXTERNLIBS}/openscenegraph

#osgearth
cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/osgEarth -DCMAKE_PREFIX_PATH=${EXTERNLIBS}/openscenegraph



#OpenCV4
cmake ..   -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/OpenCV4 -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules
disable performance tests and normal tests, build (be very patient) and install

#open62541
cmake ..  -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/open62541 -DUA_FORCE_WERROR=FALSE -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DUA_ENABLE_ENCRYPTION=OPENSSL -DUA_ENABLE_ENCRYPTION_OPENSSL=1

#ifcpp
cmake ..  -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/ifcpp -DCMAKE_PREFIX_PATH=${EXTERNLIBS}/openscenegraph/

#vsg:
cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/vsgXchange -DCMAKE_PREFIX_PATH="${EXTERNLIBS}/vsg;${EXTERNLIBS}/osg2vsg;${EXTERNLIBS}/openscenegraph;${EXTERNLIBS}/vsgXchange"

cmake .. -DCMAKE_INSTALL_PREFIX=${EXTERNLIBS}/vsgExamples  -DCMAKE_PREFIX_PATH="${EXTERNLIBS}/vsg;${EXTERNLIBS}/osg2vsg;${EXTERNLIBS}/openscenegraph;${EXTERNLIBS}/vsgXchange"

