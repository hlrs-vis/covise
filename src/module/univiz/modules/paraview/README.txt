INSTRUCTIONS FOR PARAVIEW >= 3.3
================================


INTRODUCTION
------------

Including others, Paraview offers two methods for module integration: either compiled together with the complete Paraview (BUILD_SHARED_LIBS set to OFF) referred here as "internal" compilation, or in a separate directory as shared libraries (BUILD_SHARED_LIBS set to ON) referred here as "external" or "plugin" compilation. The modules support both methods, see the appropriate section below. The modules are actually only developed and tested on the UNIX (Linux) version of Paraview, although only small changes might be necessary to run on the Windows version. See the documentation (HTML or PDF) for further limitations.


LIMITATIONS
-----------

ParaView >= 3.3 (and hence UniViz) seems to require cmake >= 2.6. This release
was tested with cmake 2.6.0. Please ignore cmake_minimum_required warnings at
this moment.


EXTERNAL COMPILATION / INSTALLATION (recommended)
-------------------------------------------------

1. Paraview has to be already compiled with BUILD_SHARED_LIBS set to ON. You
   will need the path to the directory into which Paraview has been compiled. 
2. Change to <your_path>/univiz/modules/paraview/plugins of this module distribution.
3. Generate Makefiles:
   - run ccmake (including the dot): ccmake .
   - press 'c'
   - an error will occur because ParaView_DIR is not set
   - press 'e'
   - move to the ParaView_DIR entry
   - press 'enter'
   - enter (paste) the path to the directory into which Paraview was compiled
   - press 'enter'
   - move to the CMAKE_BUILD_TYPE entry
   - press 'enter'
   - enter: Release
   - press 'enter'
   - press 'c'
   - set all the module entries (UNIVIZ_PLUGIN_*) to ON
   - press 'c'
   - ignore the messages, press 'e'
   - if UNIVIZ_PLUGIN_DumpCFX set to ON, make sure that CFX_INCLUDE_PATH and
     CFX_LIBRARY_PATH point to the right place
   - press several (0) times 'c' (until the 'g' command gets available)
   - press 'g' in order to generate the Makefiles
4. compile: make
5. installation:
   You can load the modules into Paraview by selecting the .so files inside
   this directory at the "Manage Plugins" command, *_SMPlugin for
   "Server Plugins" and *_GUIPlugin for "Client Plugins". Another possibility
   is to make the PV_PLUGIN_PATH environment variable point to
   <your_path>/univiz/modules/paraview/plugins.

   If using the DumpCFX module, make sure that the CFX file "units.cfx" is
   accessible, e.g. by putting it into the current directory when launching
   ParaView.


INTERNAL COMPILATION / INSTALLATION
-----------------------------------

1. Paraview has to be compiled with BUILD_SHARED_LIBS set to OFF.
2. Ignore ETHUnified when integrating with CSCS modules. Installing both
   ETHUnified and Univiz will lead to collisions.
3. Change to (create) the directory where Paraview was (is going to be) compiled
4. Generate Makefiles:
   - follow the configuration instructions for compiling Paraview, you will
     need at least to run: ccmake <path to source tree of Paraview>
   - press 'c'
   - press 't' to change to the extended mode
   - move to the PARAVIEW_EXTRA_EXTERNAL_MODULE entry
   - press 'enter'
   - add (separate by semicolon from existing entry): Univiz
   - press 'enter'
   - press 'c'
   - set PARAVIEW_USE_Univiz to ON
   - press 'c'
   - an error will occur because Univiz_SOURCE_DIR is not set
   - press 'e'
   - move to the Univiz_SOURCE_DIR entry
   - press 'enter'
   - enter: <your_path>/univiz/modules/paraview
   - press 'enter'
   - press 'c'
   - press 'g' in order to generate the Makefiles
5. compile: make
6. installation:
   If using the DumpCFX module, make sure that the CFX file "units.cfx" is
   accessible, e.g. by putting it into the current directory when launching
   ParaView.


SEE ALSO
--------

Documentation of the Univiz project (HTML or PDF) at univiz/doc or online at
http://graphics.ethz.ch/~sadlof/research/flowvis_modules_cgl/

http://www.vtk.org/Wiki/Plugin_HowTo
http://public.kitware.com/pipermail/paraview/2004-September/000873.html
http://www.paraview.org/Wiki/ParaView:Extend


------------------------------------------
2008-09-01, Filip Sadlo, sadlo@inf.ethz.ch
Computer Graphics Laboratory, ETH Zurich
