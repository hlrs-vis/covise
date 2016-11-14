COVISE and OpenCOVER
====================

[COVISE][1], the collaborative visualization and simulation environment, is a modular distributed visualization system.
As its focus is on visualization of scientific data in virtual environments, it comprises the VR renderer [OpenCOVER][2].
COVISE development is headed by [HLRS][3].
It is portable to Windows and UNIX. We do regular builds on x86_64 Windows, Linux and Mac OS X.

License
-------

If not stated otherwise, COVISE and OpenCOVER source code is licensed under the LGPL v2.1. See `lgpl-2.1.txt` for
details.


Support & Mailing Lists
-----------------------

As a user of COVISE, you might get answers to your questions on the [covise-users][4] mailing list.
Please direct any questions related to installing/building/using COVISE there.

You can receive notifications of changes to COVISE on the [covise-commits][5] list.


Getting COVISE
--------------

### UNIX

Getting COVISE is as easy as

      git clone https://github.com/hlrs-vis/covise.git --recursive

Update your existing copy to the current version by

      git pull -r
      git submodule sync
      git submodule update --init --recursive # update submodules to latest required version


Build Requirements
------------------

- **C++ compiler**:
  C++03 or C++11

  On Windows, only Visual Studio 2012 is tested.
- **CMake**:
  2.8.10 or newer should work
- **XercesC**:
- **Qt**:
  Either Qt 4 or 5 is required by the graphical user interface.
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
  3.2 or newer is required for the VR renderer OpenCOVER
- **Open Inventor**
  The desktop renderer on Linux requires Open Inventor.
  The binaries shipped with most Linux distributions do not correctly render fonts on 64
  bit systems.
  Install a fixed version:
        git clone https://github.com/aumuell/open-inventor.git
        cd open-inventor
        make IVPREFIX=$EXTERNLIBS/inventor install

Optional dependencies
---------------------
- **JPEG Turbo**
- **VTK**
  Version 6 is required.
- **Flex** and **Bison**
  Lexer/Parser generators, required to build VRML plugin.

CMake will show lists of met and unmet optional and required dependencies.
You should check those and install additional prerequisites as needed.


Working with Git
---------------

### UNIX
      cd covise
      git pull -r #-r requests a rebase of your changes to avoid trivial branching
      git submodule update --init --recursive # update submodules to latest required version

Building COVISE
---------------

### UNIX

      cd covise
      source .covise.sh #set environment variables
      make #invoke cmake followed by make

COVISE and OpenCOVER are built by two separate CMake projects.
This command sequence sets environment variables necessary while building
COVISE, invokes `cmake` for the COVISE project, builds COVISE, and then
continues with invoking `cmake` for the OpenCOVER project followed by `make`.

After an initial build, it is possible to invoke `make` from within
subdirectories of `covise/src`.

No installation is required: you can use COVISE directly from the build tree.

### Windows
       REM set COVISEDIR to location of your COVISE checkout
       set COVISEDIR=c:/src/covise
	   REM set EXTERNLIBS to correct location of all your dependancies
       set EXTERNLIBS=c:\src\externlibs\tamarau
       cd %COVISEDIR%
	   REM call winenv.bat with appropriate archsuffix for debug or release (tamarau for Visual Studio 2012 and zebu for 2015)
       call %COVISEDIR%\winenv.bat tamarauopt
       mkdir build.covise
       cd build.covise
       cmake-gui ..
       devenv
       cd %COVISEDIR%
       mkdir build.cover
       cd build.cover
       cmake-gui ../src/OpenCOVER
       devenv


Changing CMake Settings
-----------------------
You can influence which parts of COVISE are built by editing CMake settings in
`${COVISEDIR}/${ARCHSUFFIX}/build.covise/CMakeCache.txt`.
This might help you work around build problems.

    cd ${COVISEDIR}/${ARCHSUFFIX}/build.covise
    ccmake ../..

- `COVISE_USE_VIRVO`: disable support for direct volume rendering
- `COVISE_BUILD_DRIVINGSIM`: enable the road editor oddlot as part of the OpenCOVER CMake project
- `COVISE_USE_CUDA`: disable use of CUDA
- `COVISE_USE_QT4`: enable Qt 4
- `COVISE_CPU_ARCH`: set optimization for the CPU in your computer
- `COVISE_USE_CPP11`: disable compilation in C++11 mode
- `COVISE_BUILD_RENDERER`: disable building the desktop renderer
- `COVISE_GENERATE_DOCS`: generate HTML and PDF documentation

After changing any of these settings, you have to restart the build process.


Invoking COVISE
---------------

### UNIX

Add .../covise/bin to your PATH.

      covise


Source Code Organization
------------------------

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


[1]:   http://www.hlrs.de/covise/
[2]:   http://www.hlrs.de/solutions-services/service-portfolio/visualization/covise/opencover/
[3]:   http://www.hlrs.de/
[4]:   https://listserv.uni-stuttgart.de/mailman/listinfo/covise-users
[5]:   https://listserv.uni-stuttgart.de/mailman/listinfo/covise-commits
