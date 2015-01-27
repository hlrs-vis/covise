-=-=-=-=-=-=-=-=-=-=-
 Buildserver scripts
-=-=-=-=-=-=-=-=-=-=-

Note: the buildserver is not intended to run out of the source directory, since
during autobuild (target "nightflight") the source directory gets deleted when
TARGET.NIGHTFLIGHT.FORCECHECKOUT = true
One could fix that, but it´s tedious. Rather copy the directory structure 
described below to a location outside the COVISE sources and set it up like 
described below.

For detailed info on the VISENSO COVISE ant buildservers see
http://192.168.0.50/wiki/index.php/Vikisenso/Tools/BuildServer


-------------------
directory structure
-------------------

See the script code for explicit usage documentation.

.\Autobuild\
contains scripts used for automatically building COVISE

.\common\
contains helper scripts used by scripts of several other categories

.\manually\
contains scripts that can be executed manually to perform specific
operations, e. g. for starting Visual Studio or compiling Python

.\Setup\
contains inno setup scripts and the batches to execute the setup
generation

.\Setup\install\
contains all scripts that are to be included *AS THEY ARE* into
the COVISE setup and the COVISE shipment
especially keep an eye on the content of common.local.bat as it will
override any environment variables.
Any subdirectories of the above mentioned directory contain files
explicit to the distribution depicted by the directory´s name.


-------------------
install buildserver
-------------------
* create/adapt buildserver profile
   - create a new file uniquely named to identify your build in 
      .\Autobuild\_properties\
   - override any settings inside of .\Autobuild\default.properties
      that you want changed by copying that setting to your new file
      and changing it in your new file (not in default.properties!)
   - copy your new file to .\Autobuild\build.properties (possibly
      copying it over a file existing by the same name)
   - create a file .\Autobuild\ShipmentARCHSUFFIXes.txt and enter
      all the ARCHSUFFIXes for which shipments are to be build, one
      ARCHSUFFIX per line. sample file content can be e. g.:
vistaopt
vista
      Then both shipments will be created: the optimized and the
      non-optimized version of the 'vista'-distribution.
* copy some necessary files from the COVISE sources to the build server location
   - copy %COVISEDIR%\common.local.bat to .\common\
   - copy %COVISEDIR%\common.VISENSO.bat to .\common\
   - copy %COVISEDIR%\get8dot3path.vbs to .\common\
   This is necessary since the autobuildserver deletes the COVISE source
   directory during a nightly build completely.
* adapt hardcoded paths in scripts
   - adapt all settings necessary for execution of the local 
      buildserver installation in the file .\common\common.local.bat
      This file should at least have the local EXTERNLIBS defined.
      If COCONFIG is set here, do not forget, when later on executing
      COVISE.
* put the COVISE licenses (files "config.license.xml" and/or "config-license.xml")
   for the local computer into the directory .\..\licenses\


-------------------
more things to do
-------------------
   - you can use the batches .\manually\startVCvista.bat and
      .\manually\startVCvistaopt.bat (or similar for other ARCHSUFFIXes)
      to start a Visual C++ IDE configured for the supplied ARCHSUFFIX
      Assure that you have the path to the Visual C++ batch vcvars.bat
      added to PATH (e. g. you can do this in common.local.bat) like
      SET PATH=%PATH%;C:\Programme\Microsoft Visual Studio 8\VC
   - if you like to compile numpy on a 64bit system, the file
      .\manually\compileNumpy_x64.bat has to be adapted (also see the
      notes in the Vikisenso)
   - adapt the local paths in .\manually\VCbuild_eval.bat then you can 
      copynpaste the output of a Visual Studio C++ compile into the file 
      .\manually\VCbuild_temp.txt and then execute .\manually\VCbuild_eval.bat
      This will generate a textfile only containing the errors lines of the 
      compile output.