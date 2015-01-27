INSTRUCTIONS FOR PARAVIEW < 3
=============================


INTRODUCTION
------------

Including others, Paraview offers two methods for module integration: either compiled together with the complete Paraview (BUILD_SHARED_LIBS set to OFF) referred here as "internal" compilation, or in a separate directory as shared libraries (BUILD_SHARED_LIBS set to ON) referred here as "external" compilation. The modules support both methods, see the appropriate section below. The modules are actually only developed and tested on the UNIX (Linux) version of Paraview, although only small changes might be necessary to run on the Windows version. See the documentation (HTML or PDF) for further limitations.


EXTERNAL COMPILATION / INSTALLATION
-----------------------------------

For each module do:

1. Paraview has to be already compiled with BUILD_SHARED_LIBS set to ON. You
   will need the path to the directory into which Paraview has been compiled. 
2. Change to the extracted directory of this module distribution.
3. If using Paraview 2.4, rename PVLocal.pvsm.in.paraview_2.4 to PVLocal.pvsm.in
4a. either edit <your_path>/univiz/modules/paraview/Makefile and do 'make' or:
4b. Generate Makefiles
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
   - press several (2) times 'c' (until the 'g' command gets available)
   - press 'g' in order to generate the Makefiles
   compile: make
5. installation:
   You can load the module into Paraview by selecting the .xml file inside
   this directory at the "Load Package" command. Another possibility is to
   copy the XML file together with the .so files to a directory pointed to
   by PV_INTERFACE_PATH.


INTERNAL COMPILATION / INSTALLATION
-----------------------------------

For all modules together do once:

1. Paraview has to be compiled with BUILD_SHARED_LIBS set to OFF.
2. Ignore ETHUnified when integrating with CSCS modules. Installing both
   ETHUnified and Univiz will lead to collisions.
3. Change to (create) the directory where Paraview was (is going to be) compiled
4. If using Paraview 2.4, rename Univiz_Server.xml.paraview_2.4 to Univiz_Server.xml
5. Generate Makefiles
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
6. compile: make


SEE ALSO
--------

Documentation of the Univiz project (HTML or PDF) at univiz/doc or online at
http://graphics.ethz.ch/~sadlof/research/flowvis_modules_cgl/

http://public.kitware.com/pipermail/paraview/2004-September/000873.html
http://www.paraview.org/Wiki/ParaView:Extend


------------------------------------------
2007-05-24, Filip Sadlo, sadlo@inf.ethz.ch
Computer Graphics Laboratory, ETH Zurich
