Welcome to DeskVOX Volume Explorer (DeskVOX)

0. Content
==========

 1. How to obtain DeskVOX
 2. Additional dependencies
 3. How to build DeskVOX and vconv from source
 4. How to build the Virvo library from source

***

1. How to obtain DeskVOX
========================

Download a zip archive from [Github](https://github.com/deskvox/deskvox)
or check out DeskVOX via Git:

    $ git clone https://github.com/deskvox/deskvox.git


2. Additional dependencies
==========================

The Virvo library won't build without the following libraries installed

- Pthreads: POSIX Threads Library
- GLEW: The [OpenGL Extension Wrangler Library](http://glew.sourceforge.net)
- Boost: [Boost C++ Libraries](http://www.boost.org)

The new DeskVOX GUI depends on Qt, either version 4 or 5:

- [Qt Toolkit](http://qt-project.org): GUI

The old DeskVOX application won't build without the following libraries installed

- [FOX TOOLKIT 1.6](http://www.fox-toolkit.org): GUI

Having the following libraries installed is recommended but not necessary:

- [Cg Toolkit](http://developer.nvidia.com/cg-toolkit) (NVIDIA COORP)
- [CUDA Toolkit](http://developer.nvidia.com/category/zone/cuda-zone) (NVIDIA COORP)
- [FFmpeg](http://ffmpeg.org/): video codecs
- [snappy](http://code.google.com/p/snappy/): fast compression

Not all features will be available without these libraries.


3. How to build DeskVOX and vconv from source
=============================================

DeskVOX uses the [CMake](http://www.cmake.org/) build system to generate a
project specific to your platform to be built from. Obtain CMake from
[http://www.cmake.org](http://www.cmake.org/) or by means
provided by your operating system.

Switch to the topmost folder of the DeskVOX package.

CMake encourages you to do so called out-of-source builds, i.e. all files, e.g.
object files, executables or auto-generated headers will be located in a folder
separate from the folder the source files are located in.

In order to perform an out-of-source build, create a new build directory, e.g.:

    $ mkdir build

Change to that directory:

    $ cd build

Invoke CMake:

    $ cmake ..

This will generate a project files suitable for your platform, e.g. Visual
Studio Solutions on Windows or a Makefile project on Unix.
Edit CMakeCache.txt to specify custom paths to additional libraries, perform
a Debug build, etc. Alternatively, use `ccmake` or a CMake GUI instead
of `cmake`.

After CMake has generated the build files for your build system, proceed
as required.

On Unix platforms, type:

    $ make

After this step, the DeskVOX application will be located in `build/vox-desk/src`,
the static Virvo library in `build/virvo/virvo`,
the vview test application in `build/virvo/tools/vview`
and the vconv conversion tool in `build/virvo/tools/vconv`.

In order to install DeskVOX and its associated files type:

    $ make install

DeskVOX will be installed to a default location, which can be modified
before installing by editing `CMakeCache.txt` in your build directory.

On Windows, use the generated solution files for Microsoft Visual Studio
and build the corresponding targets in the IDE.
