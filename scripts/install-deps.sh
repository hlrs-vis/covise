#! /bin/bash

ROOT=sudo

case "${ARCHSUFFIX%opt}" in
    macos)
        cd $COVISEDIR && brew bundle
        ;;
    rhel7)
        #$ROOT yum install -y http://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-7-12.noarch.rpm 
        $ROOT yum update -y && \
        $ROOT yum -y install epel-release && \
        $ROOT yum -y install git cmake3 make ninja-build gcc-c++ && \
        $ROOT yum -y install tbb-devel libXmu-devel libXi-devel boost169-devel && \
        $ROOT yum -y install which python36-devel && \
        $ROOT yum -y install qt5-qttools-devel qt5-qtwebkit-devel qt5-qtbase-devel qt5-qtsvg-devel qt5-qtscript-devel && \
        $ROOT yum -y install qt5-qttools-static python36-pyqt5-sip \
            zlib-devel glew-devel libtiff-devel libpng-devel libjpeg-turbo-devel \
            xerces-c-devel
        ;;
    rhel8)
        #$ROOT yum install -y http://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-7-12.noarch.rpm 
        $ROOT yum update -y && \
        $ROOT yum -y install epel-release && \
        $ROOT yum --enablerepo PowerTools -y install git cmake make ninja-build gcc-c++ && \
        $ROOT yum -y install tbb-devel libXmu-devel libXi-devel boost-devel && \
        $ROOT yum -y install which python3-devel && \
        $ROOT yum -y install qt5-qttools-devel qt5-qtwebkit-devel qt5-qtbase-devel qt5-qtsvg-devel qt5-qtscript-devel && \
        $ROOT yum --enablerepo PowerTools -y install qt5-qttools-static python3-pyqt5-sip \
            zlib-devel glew-devel libtiff-devel libpng-devel libjpeg-turbo-devel \
            xerces-c-devel
        ;;
    xerus|xenial|bionic|focal|jammy|noble)
        $ROOT apt update && \
        $ROOT apt install \
            cmake ninja-build make git swig flex bison g++ \
            libxerces-c-dev \
            qttools5-dev qtscript5-dev libqt5scripttools5 libqt5svg5-dev libqt5opengl5-dev libqt5webkit5-dev \
            libboost-dev libboost-filesystem-dev libboost-serialization-dev \
            libboost-chrono-dev libboost-program-options-dev libboost-thread-dev \
            libboost-regex-dev libboost-iostreams-dev libboost-locale-dev \
            libglew-dev libice-dev libopenscenegraph-dev \
            libjpeg-dev libpng-dev libtiff-dev \
            python3-pyqt5 pyqt5-dev pyqt5-dev-tools qt5-default

        ;;
    *)
        echo "don't know how to install COVISE dependencies for ARCHSUFFIX='${ARCHSUFFIX}'"
        exit 1
        ;;
esac
