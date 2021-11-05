[![Build Status](https://travis-ci.org/hlrs-vis/covise.svg?branch=master)](https://travis-ci.org/hlrs-vis/covise)

Table of Contents
=================

* [Software in This Repository](#software-in-this-repository)
   * [COVISE and OpenCOVER](#covise-and-opencover)
   * [Other Software](#other-software)
* [License](#license)
* [Support &amp; Mailing Lists](#support--mailing-lists)
* [Getting Binaries and Automated Installation](#getting-binaries-and-automated-installation)
   * [macOS](#macos)
   * [Windows](#windows)
* [Getting the Source Code](#getting-the-source-code)
   * [UNIX and Command Line](#unix-and-command-line)
   * [Working with Git](#working-with-git)
      * [UNIX](#unix)
* [Building COVISE and OpenCOVER](#building-covise-and-opencover)
   * [Build Requirements](#build-requirements)
   * [Optional Dependencies](#optional-dependencies)
   * [Tracking Dependencies](#tracking-dependencies)
   * [Directory for Third Party Dependencies](#directory-for-third-party-dependencies)
   * [Building on UNIX](#building-on-unix)
      * [Building only OpenCOVER](#building-only-opencover)
   * [Building on Windows](#building-on-windows)
   * [Changing CMake Settings](#changing-cmake-settings)
* [Building Documentation](#building-documentation)
* [Invoking COVISE](#invoking-covise)
   * [UNIX](#unix-1)
   * [Windows](#windows-1)
* [Source Code Organization](#source-code-organization)



Software in This Repository
===========================

COVISE and OpenCOVER
--------------------

[COVISE][1], the collaborative visualization and simulation environment, is a modular distributed visualization system.
As its focus is on visualization of scientific data in virtual environments, it comprises the VR renderer [OpenCOVER][2].
COVISE development is headed by [HLRS][3].
It is portable to Windows and UNIX. We do regular builds on x86_64 Windows, Linux and macOS.

Other Software
--------------
Also included in this repository are [OddLOT][8], an OpenDRIVE editor, as well as vrmlExp, a VRML97 and X3D exporter for Autodesk 3ds Max.

License
=======

If not stated otherwise, the source code in this repository is licensed under the LGPL v2.1.
See `lgpl-2.1.txt` for details.


Support & Mailing Lists
=======================

As a user of COVISE, you might get answers to your questions on the [covise-users][4] mailing list.
Please direct any questions related to installing/building/using COVISE there.

You can receive notifications of changes to COVISE on the [covise-commits][5] list.


Getting Binaries and Automated Installation
===========================================

If you need OddLOT only, please have a look at [https://github.com/hbanzhaf/docker_covise][9].

macOS
-----

There is a [Homebrew][6] formula for COVISE. If you have it on your system, then you can simply

      brew install hlrs-vis/tap/covise

This will install COVISE, OpenCOVER, and OddLOT with all their dependencies.


Windows
-------

Windows binaries, which include COVISE, OpenCOVER and OddLOT, can be found on the [COVISE download page][7].
A separate installer for the VRML exporter vrmlExp is also available on the same webpage.


Getting the Source Code
=======================

UNIX and Command Line
---------------------

Getting COVISE is as easy as

      git clone https://github.com/hlrs-vis/covise.git --recursive

Update your existing copy to the current version by

      git pull -r
      git submodule sync
      git submodule update --init --recursive # update submodules to latest required version




Working with Git
---------------

### UNIX
      cd covise
      git pull -r #-r requests a rebase of your changes to avoid trivial branching
      git submodule update --init --recursive # update submodules to latest required version


Building COVISE and OpenCOVER
=============================

Build Requirements
------------------

The script `scripts/install-deps.sh` will help you to install the dependencies
provided by your Linux distribution (Debian/Ubuntu and RHEL/CentOS).

- **C++ compiler**:
  C++11

  On Windows, we currently use Visual Studio 2019 (VC15).
  GCC 5.3 and newer should work.

- **CMake**:
  3.1 or newer is required, but currently we suggest CMake 3.7 or newer
- **XercesC**:
- **Qt**:
  Qt 5 is required by the graphical user interface.
  If you want to use the Qt/Coin3D/SoQt based desktop renderer (QtRender),
  then this version of Qt has to match the one that SoQt is built against.

  For Qt5, you need the following modules:
    - `Qt5Core`
    - `Qt5Network`
    - `Qt5Xml`
    - `Qt5Widgets`
    - `Qt5OpenGL`
    - `Qt5WebKit`
    - `Qt5WebKitWidgets`
    - `Qt5Gui`
    - `Qt5Svg`
    - `Qt5PrintSupport`
    - `Qt5UiTools`
    - `Qt5Script`
    - `Qt5ScriptTools`

  On Ubuntu 14.04, you should be able to install the required packages with
  this command:
  `sudo apt-get install qttools5-dev qtscript5-dev libqt5scripttools5 libqt5svg5-dev libqt5opengl5-dev libqt5webkit5-dev`

- **Boost**:
  1.52 and newer should work, following boost libraries are required:
    - `chrono`
    - `date-time`
    - `filesystem`
    - `iostreams`
    - `locale`
    - `program-options`
    - `regex`
    - `serialization`
    - `system`
    - `thread`
  When any of these are missing, you will only get a generic message, that "Boost" is missing. Thus beware!
  Ubuntu 14.04: `sudo apt-get install libboost-all-dev`
- **Python**:
  Python 3 is required for the GUI vr-prepare and for the scripting interface
- **GLEW**:
  Used for OpenGL extension handling in Virvo (direct volume rendering) and OpenCOVER
- **OpenSceneGraph**:
  3.2 or newer is required, 3.4 or newer highly recommended for the VR and desktop renderer OpenCOVER


Optional Dependencies
---------------------
- **JPEG Turbo**
- **VTK**
  Version 6 or newer is required.
- **Flex** and **Bison**
  Lexer/Parser generators, required to build VRML plugin.
- **Open Inventor**
  The old desktop renderer on Linux requires Open Inventor.
  The binaries shipped with most Linux distributions do not correctly render fonts on 64
  bit systems.
  Install a fixed version:

        git clone https://github.com/aumuell/open-inventor.git
        cd open-inventor
        cmake -DCMAKE_INSTALL_PREFIX=$EXTERNLIBS/inventor ..
        cmake --build . --target all
        cmake --build . --target install

Tracking Dependencies
---------------------
CMake will show lists of met and unmet optional and required dependencies.
You should check those and install additional prerequisites as needed.


Directory for Third Party Dependencies
--------------------------------------
COVISE is set up to automatically search for third party libraries in
subdirectories of a directory pointed to by the environment variable
EXTERNLIBS.
You should install e.g. OpenSceneGraph into $EXTERNLIBS/openscenegraph, and
it will be discovered during the build process.

Building on UNIX
----------------

      cd covise
      source .covise.sh #set environment variables
      make #invoke cmake followed by make

This command sequence sets environment variables necessary while building
COVISE, invokes `cmake` for the COVISE project, and builds COVISE with 
OpenCOVER.

After an initial build, it is possible to invoke `make` from within
subdirectories of `covise/src`.

No installation is required: you can use COVISE directly from the build tree.

### Building only OpenCOVER

      cd covise
      source .covise.sh #set environment variables
      COVISE_CMAKE_OPTIONS=-DCOVISE_BUILD_ONLY_COVER=ON make #invoke cmake with provided options followed by make

### Building only OddLOT

      cd covise
      source .covise.sh #set environment variables
      COVISE_CMAKE_OPTIONS=-DCOVISE_BUILD_ONLY_ODDLOT=ON make #invoke cmake with provided options followed by make

Building on Windows
-------------------
Also on Windows, you should work from a command prompt:

       REM set COVISEDIR to location of your COVISE checkout
       set COVISEDIR=c:/src/covise
	   REM set EXTERNLIBS to correct location of all your dependancies
       set EXTERNLIBSROOT=c:\src\externlibs
       cd %COVISEDIR%
	   REM call winenv.bat with appropriate archsuffix for debug or release (tamarau for Visual Studio 2012 and zebu for 2015)
       call %COVISEDIR%\winenv.bat zebuopt
       mkdir build.covise
       cd build.covise
       cmake-gui ..
       REM open Visual Studio - either directly or with the button from CMake GUI
       devenv


Changing CMake Settings
-----------------------
You can influence which parts of COVISE are built by editing CMake settings in
`${COVISEDIR}/${ARCHSUFFIX}/build.covise/CMakeCache.txt`.
This might help you work around build problems.

    cd ${COVISEDIR}/${ARCHSUFFIX}/build.covise
    ccmake ../..

- `COVISE_BUILD_ONLY_COVER`: build only the OpenCOVER VR/desktop renderer without the COVISE visualization pipeline
- `COVISE_BUILD_ONLY_ODDLOT`: build only the road editor OddLOT
- `COVISE_BUILD_DRIVINGSIM`: enable the driving simulator components of OpenCOVER
- `COVISE_USE_VIRVO`: disable support for direct volume rendering
- `COVISE_USE_CUDA`: disable use of CUDA
- `COVISE_CPU_ARCH`: set optimization for the CPU in your computer
- `COVISE_BUILD_RENDERER`: disable building the desktop renderer

After changing any of these settings, you have to restart the build process.

You can also provide initial CMake options by adding them to the environment `COVISE_CMAKE_OPTIONS` before calling `make`.

Building Documentation
======================

COVISE retrieves documentation from the web server at HLRS.
But you also can build the documentation locally. You need the following
tools:

- pdflatex
- latex2html
- doxygen
- graphviz

Then you can:

    cd ${COVISEDIR}/doc
    make


Invoking COVISE
===============

UNIX
----

Add .../covise/bin to your PATH.

      covise

Windows
-------

COVISE can be used without installation, provided you take the same steps as
for building:

       REM set COVISEDIR to location of your COVISE checkout
       set COVISEDIR=c:/src/covise
	   REM set EXTERNLIBS to correct location of all your dependancies
       set EXTERNLIBSROOT=c:\src\externlibs
       cd %COVISEDIR%
	   REM call winenv.bat with appropriate archsuffix for debug or release (tamarau for Visual Studio 2012 and zebu for 2015 Update 3, 2017 or 2019)
       call %COVISEDIR%\winenv.bat zebuopt
       covise
       opencover


Source Code Organization
========================

- `cmake`:
  cmake files

- `doc`:
  documentation and tools for creating documentation

- `config`:
  configuration examples

- `scripts`:
  support scripts for building COVISE

- `share`:
  architecture independent files: textures, shaders, example data, ...

- `src`:
  source code

  - `src/3rdparty`:
    3rd party source code

  - `src/tools`:
    various programs related to building or using COVISE

  - `src/kernel`:
    COVISE core libraries

  - `src/sys`:
    COVISE core executables

  - `src/module`:
    COVISE visualization modules (algorithms)

  - `src/OpenCOVER`:
    VR renderer with its plug-ins

  - `src/oddlot`:
    OpenDRIVE road editor OddLOT


[1]:   http://www.hlrs.de/covise/
[2]:   http://www.hlrs.de/solutions-services/service-portfolio/visualization/covise/opencover/
[3]:   http://www.hlrs.de/
[4]:   https://listserv.uni-stuttgart.de/mailman/listinfo/covise-users
[5]:   https://listserv.uni-stuttgart.de/mailman/listinfo/covise-commits
[6]:   http://brew.sh
[7]:   https://fs.hlrs.de/projects/covise/support/download/
[8]:   http://www.hlrs.de/oddlot/
[9]:   https://github.com/hbanzhaf/docker_covise
