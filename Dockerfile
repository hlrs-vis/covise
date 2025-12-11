# Example with podman/nvidia container toolkit/X11:
# podman run --rm -it --device nvidia.com/gpu=all \
#        -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro covise:latest \
#         /app/bin/opencover

FROM quay.io/rockylinux/rockylinux:9.7

# Minimum packages to build COVISE on Rocky Linux 9,
# Install libLAS from EPEL Testing to fix dependency issues with OpenSceneGraph
RUN dnf install -y dnf-plugins-core epel-release && \
    dnf config-manager --set-enabled crb && \
    dnf clean all && \
    dnf install -y cmake git make which \
    boost-devel gcc g++ qt6-qtbase-devel qt5-qtbase-devel \
    qt6-qtsvg-devel qt6-qtpositioning-devel \
    qt6-qtlocation-devel qt6-qttools-devel qt6-qtbase-private-devel glew-devel \
    openssl-devel xerces-c-devel libjpeg-turbo-devel \
    libpng-devel libtiff-devel gdal-devel libzstd-devel bzip2-devel && \
    dnf --enablerepo=epel-testing install liblas-devel -y && \ 
    dnf install -y OpenSceneGraph-devel && \
    dnf clean all

COPY . /app/

WORKDIR /app/

RUN bash -c "source .covise.sh && make -j$(nproc)"
