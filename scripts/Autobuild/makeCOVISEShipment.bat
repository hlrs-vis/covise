@ECHO OFF

SETLOCAL ENABLEDELAYEDEXPANSION

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM ******************************************
REM Deploys a COVISE shipment
REM ******************************************
ECHO Will deploy a COVISE shipment.
ECHO Starting...





REM ****************************
REM Set environment variables
REM ****************************

REM Establish command aliases and constants
REM SET OPCOPY=XCOPY /K /V /I /R /Q /Y /C
SET OPCOPY=COPY /V /Y
SET OPCOPYDIR=XCOPY /E /V /I /R /Q /Y /C
SET OPMKDIR=MD
SET OPDELDIR=RD /S /Q
SET OPRENAME=REN
SET OPDEL=DEL /Q

REM SET OPPYUIC=%PYTHONHOME%\pyuic4.bat -w
REM SET OPPYUIC_D=%PYTHONHOME%\pyuic4_d.bat -w

SET OPPYRCC=%PYTHONHOME%\pyrcc4.exe
SET OPPYRCC_D=%PYTHONHOME%\pyrcc4_d.exe

SET OPEXIST=IF EXIST
SET OPNEXIST=IF NOT EXIST
SET FILE_ARCHSUFFIXES=ShipmentARCHSUFFIXes.txt
SET FILETEMP_LEFT="%TMP%\left.txt"
SET FILETEMP_RIGHT="%TMP%\right.txt"
SET FILETEMP_DIFF="%TMP%\diff.txt"
SET FILETEMP_DIFFCNT="%TMP%\diffcnt.txt"
SET FILETEMP_UIFILES="%TMP%\uifiles.txt"
SET FILETEMP_SUBDIRS="%TMP%\subfolders.txt"
SET FILETEMP_UIDUMMY="%TMP%\~tmp.ui"
SET FILETEMP_PYDUMMY="%TMP%\~tmp.py"
SET FLAG_POSTPROCESS_ARCSUF=

IF "x%1x" EQU "xx" GOTO USAGE
IF "x%2x" EQU "xx" GOTO USAGE
IF "x%3x" EQU "xx" GOTO USAGE
IF "x%4x" EQU "xx" GOTO USAGE
SET INSTALLTARGET=%5
IF "x%INSTALLTARGET%x" EQU "xx" SET INSTALLTARGET=%ARCHSUFFIX%

SET SRCDIR=%1
SET DESTDIR=%2
SET COVISEDIR=%2\covise
SET PATH=%PATH%;%COVISEDIR%
SET COMSRCDIR=%1\..\common
SET RESSRCDIR=%1
SET LICENSEFILE=%3
IF "x%LICENSEFILE%x"=="xx" SET LICENSEFILE=%RESSRCDIR%\config
SET UNIXUTILS=%4

REM get the 8.3 filename of current installation
SET TMPFILE="%TMP%\~covtmpfile.txt"
CALL "%WINDIR%\system32\cscript.exe" /nologo "%SRCDIR%\get8dot3path.vbs" "%~dp0">%TMPFILE%
SET /P INSTALLDIR=<%TMPFILE%
DEL /Q %TMPFILE%

ECHO ... ARCHSUFFIX=%ARCHSUFFIX%
ECHO ... BUILDDIR=%BUILDDIR%
ECHO ... INSTALLDIR=%INSTALLDIR%
ECHO ... SRCDIR=%SRCDIR%
ECHO ... DESTDIR=%DESTDIR%
ECHO ... RESSRCDIR=%RESSRCDIR%
ECHO ... EXTERNLIBS=%EXTERNLIBS%
ECHO ... COVISEDIR=%COVISEDIR%
ECHO ... LICENSEFILE=%LICENSEFILE%
ECHO ... PTHONHOME=%PYTHONHOME%
ECHO ... UNIXUTILS=%UNIXUTILS%

IF NOT EXIST %INSTALLDIR%\%FILE_ARCHSUFFIXES% GOTO ERROR_NOARCHSUFFIXFILE





REM ******************************************************************
REM Perform operations common for all ARCHSUFFIXes
REM ******************************************************************

ECHO ... deleting possibly existing shipment data in destination dir first ...
REM %OPDELDIR% %DESTDIR%

REM ECHO ... filling %DESTDIR%\common\bin ...
%OPMKDIR% %DESTDIR%\common\bin
%OPCOPY% %COMSRCDIR%\bin\* %DESTDIR%\common\bin > NUL






REM ******************************************************************
REM assemble files for each ARCHSUFFIX independently
REM ******************************************************************
ECHO ... reading ARCHSUFFIXes from file %INSTALLDIR%\%FILE_ARCHSUFFIXES% ...

FOR /F %%A IN (%INSTALLDIR%\%FILE_ARCHSUFFIXES%) DO (

   ECHO ... now assembling ARCHSUFFIX=%%A ...

   REM %OPEXIST% %COVISEDIR%\%%A %OPDELDIR% %COVISEDIR%\%%A

   REM remember the first ARCHSUFFIX to be the one for which
   REM the postprocessing will take place later on
   IF "x!FLAG_POSTPROCESS_ARCSUF!x" EQU "xx" SET FLAG_POSTPROCESS_ARCSUF=%%A





   REM *******************
   REM copy the files
   REM *******************


   REM ----------------
   REM copy executables
   REM ----------------
   
   ECHO ... filling %COVISEDIR%\%%A\bin and subdirs ...
   %OPMKDIR% %COVISEDIR%\%%A\bin
   %OPCOPY% %SRCDIR%\%%A\bin\*.exe %COVISEDIR%\%%A\bin\ > NUL
   REM %OPCOPY% %SRCDIR%\%%A\bin\*.bat %COVISEDIR%\%%A\bin\ > NUL

   REM %OPMKDIR% %COVISEDIR%\%%A\bin\Color
   REM %OPCOPY% %SRCDIR%\%%A\bin\Color\*.exe %DESTDIR%\%%A\bin\Color\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Converter
   %OPCOPY% %SRCDIR%\%%A\bin\Converter\*.exe %COVISEDIR%\%%A\bin\Converter\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Develop
   %OPCOPY% %SRCDIR%\%%A\bin\Develop\*.exe %COVISEDIR%\%%A\bin\Develop\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Examples
   %OPCOPY% %SRCDIR%\%%A\bin\Examples\*.exe %COVISEDIR%\%%A\bin\Examples\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Filter
   %OPCOPY% %SRCDIR%\%%A\bin\Filter\*.exe %COVISEDIR%\%%A\bin\Filter\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Interpolator
   %OPCOPY% %SRCDIR%\%%A\bin\Interpolator\*.exe %COVISEDIR%\%%A\bin\Interpolator\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\IO_Module
   %OPCOPY% %SRCDIR%\%%A\bin\IO_Module\*.exe %COVISEDIR%\%%A\bin\IO_Module\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Mapper
   %OPCOPY% %SRCDIR%\%%A\bin\Mapper\*.exe %COVISEDIR%\%%A\bin\Mapper\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Obsolete
   %OPCOPY% %SRCDIR%\%%A\bin\Obsolete\*.exe %COVISEDIR%\%%A\bin\Obsolete\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Renderer
   %OPCOPY% %SRCDIR%\%%A\bin\Renderer\*.exe %COVISEDIR%\%%A\bin\Renderer\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\SCA
   %OPCOPY% %SRCDIR%\%%A\bin\SCA\*.exe %COVISEDIR%\%%A\bin\SCA\ > NUL

   REM %OPMKDIR% %COVISEDIR%\%%A\bin\Shader
   REM %OPCOPY% %SRCDIR%\%%A\bin\Shader\*.exe %COVISEDIR%\%%A\bin\Shader\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Simulation
   %OPCOPY% %SRCDIR%\%%A\bin\Simulation\*.exe %COVISEDIR%\%%A\bin\Simulation\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Tools
   %OPCOPY% %SRCDIR%\%%A\bin\Tools\*.exe %COVISEDIR%\%%A\bin\Tools\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Tracer
   %OPCOPY% %SRCDIR%\%%A\bin\Tracer\*.exe %COVISEDIR%\%%A\bin\Tracer\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\UnderDev
   %OPCOPY% %SRCDIR%\%%A\bin\UnderDev\*.exe %COVISEDIR%\%%A\bin\UnderDev\ > NUL

   %OPMKDIR% %COVISEDIR%\%%A\bin\Unsupported
   %OPCOPY% %SRCDIR%\%%A\bin\Unsupported\*.exe %COVISEDIR%\%%A\bin\Unsupported\ > NUL

   REM --------------
   REM copy libraries
   REM --------------
   ECHO ... filling %COVISEDIR%\%%A\lib and subdirs ...
   %OPMKDIR% %COVISEDIR%\%%A\lib\
   %OPCOPY% %SRCDIR%\%%A\lib\*.dll %COVISEDIR%\%%A\lib\ > NUL
   REM %OPCOPY% %SRCDIR%\%%A\lib\*.lib %COVISEDIR%\%%A\lib\ > NUL
   %OPMKDIR% %COVISEDIR%\%%A\lib\OpenCOVER\plugins
   %OPCOPY% %SRCDIR%\%%A\lib\OpenCOVER\plugins\*.dll %COVISEDIR%\%%A\lib\OpenCOVER\plugins\ > NUL
   REM %OPCOPY% %SRCDIR%\%%A\lib\OpenCOVER\plugins\*.lib %COVISEDIR%\%%A\lib\OpenCOVER\plugins\ > NUL
   %OPMKDIR% %COVISEDIR%\%%A\lib\sgplugins
   %OPCOPY% %SRCDIR%\%%A\lib\sgplugins\*.dll %COVISEDIR%\%%A\lib\sgplugins\ > NUL

   REM --------------
   REM copy libraries
   REM --------------
   ECHO ... copying Python scripts from %SRCDIR%\%%A\lib ...
   %OPCOPY% %SRCDIR%\%%A\lib\*.py %COVISEDIR%\Python\ > NUL





   REM -----------------------------
   REM postprocessing per ARCHSUFFIX
   REM -----------------------------
   IF "x%%A:~-3,3%x" EQU "xoptx" (
      REM optimized version
      SET COVISEPYD=_covise
      SET COGRMSGPYD=_coGRMsg
   ) ELSE (
      REM debug version
      SET COVISEPYD=_covise_d
      SET COGRMSGPYD=_coGRMsg_d
   )
   %OPCOPY% %SRCDIR%\%%A\lib\!COVISEPYD!.pyd %COVISEDIR%\Python\ > NUL
   REM the following means: if the ERRORLEVEL is >= than 0
   IF ERRORLEVEL 1 (
      ECHO ... %SRCDIR%\%%A\lib\!COVISEPYD!.pyd not found; trying other filename ^(%SRCDIR%\%%A\lib\!COVISEPYD!.dll^)...
      %OPCOPY% %SRCDIR%\%%A\lib\!COVISEPYD!.dll !COVISEDIR!\Python\ > NUL
      %OPRENAME% %COVISEDIR%\Python\!COVISEPYD!.dll !COVISEPYD!.pyd
   )
   %OPCOPY% %SRCDIR%\%%A\lib\coGRMsg.py %COVISEDIR%\Python\ > NUL
   %OPCOPY% %SRCDIR%\%%A\lib\!COGRMSGPYD!.pyd %COVISEDIR%\Python\ > NUL
)
IF "x%FLAG_POSTPROCESS_ARCSUF%x" EQU "xx" GOTO ERROR_ARCSUFFILEEMPTY

%OPEXIST% %INSTALLDIR%\..\Setup\install\%INSTALLTARGET%\common.local.bat %OPCOPY% %INSTALLDIR%\..\Setup\install\%INSTALLTARGET%\common.local.bat %COVISEDIR% > NUL




REM --------------
REM copy resources
REM --------------
ECHO ... copying resources ...
%OPMKDIR% %COVISEDIR%\config\
%OPCOPYDIR% %RESSRCDIR%\config %COVISEDIR%\config > NUL
%OPCOPY% %LICENSEFILE%\config.license.xml %COVISEDIR%\config\ > NUL
%OPCOPY% %LICENSEFILE%\config-license.xml %COVISEDIR%\config\ > NUL
SET COCONFIG=%COVISEDIR%\config\config.xml
%OPMKDIR% %COVISEDIR%\Python\
%OPCOPYDIR% %RESSRCDIR%\Python %COVISEDIR%\Python > NUL

%OPMKDIR% %COVISEDIR%\bitmaps\
%OPCOPYDIR% %RESSRCDIR%\bitmaps %COVISEDIR%\bitmaps > NUL
   REM %OPMKDIR% %COVISEDIR%\data\
   REM %OPCOPYDIR% %RESSRCDIR%\data %COVISEDIR%\data > NUL
%OPMKDIR% %COVISEDIR%\fonts\
%OPCOPYDIR% %RESSRCDIR%\fonts %COVISEDIR%\fonts > NUL
%OPMKDIR% %COVISEDIR%\icons\
%OPCOPYDIR% %RESSRCDIR%\icons %COVISEDIR%\icons > NUL
%OPMKDIR% %COVISEDIR%\materials\
%OPCOPYDIR% %RESSRCDIR%\materials %COVISEDIR%\materials > NUL
%OPMKDIR% %COVISEDIR%\CgPrograms\
%OPCOPYDIR% %RESSRCDIR%\CgPrograms %COVISEDIR%\CgPrograms > NUL
   REM %OPCOPY% %RESSRCDIR%\* %COVISEDIR% > NUL
%OPMKDIR% %COVISEDIR%\src\Visenso\branches\pyqt4\
%OPCOPYDIR% %SRCDIR%\src\Visenso\branches\pyqt4 %COVISEDIR%\src\Visenso\branches\pyqt4 > NUL
REM %OPMKDIR% %COVISEDIR%\src\Visenso\ui\
REM %OPCOPYDIR% %SRCDIR%\src\Visenso\ui %COVISEDIR%\src\Visenso\ui > NUL
%OPCOPY% %SRCDIR%\README.windows %COVISEDIR%\README.txt > NUL
%OPCOPY% %SRCDIR%\rgb.txt %COVISEDIR% > NUL





REM --------------
REM postprocessing
REM --------------

ECHO ... general postprocessing ...
%OPCOPY% %SRCDIR%\src\sys\ScriptingInterface\covise.py %COVISEDIR%\Python\ > NUL
IF "%ERRORLEVEL%" NEQ "0" (
   ECHO ... new covise.py not found; trying versioned one ...
   %OPCOPY% %SRCDIR%\src\sys\ScriptingInterface\win32\covise.py %COVISEDIR%\Python\ > NUL   
)
%OPCOPY% %SRCDIR%\common.bat %COVISEDIR%\ > NUL
%OPEXIST% %SRCDIR%\mycommon.bat %OPCOPY% %SRCDIR%\mycommon.bat %COVISEDIR%\ > NUL
%OPCOPY% %SRCDIR%\..\common\bin\common-base.bat %DESTDIR%\common\bin\ > NUL

REM Note: %FLAG_POSTPROCESS_ARCSUF% has been set above to the first ARCHSUFFIX
REM    found in the file %INSTALLDIR%\%FILE_ARCHSUFFIXES%
ECHO ... Python binaries will be compiled for ARCHSUFFIX=%FLAG_POSTPROCESS_ARCSUF% ...
CALL %COVISEDIR%\common.VISENSO.bat %FLAG_POSTPROCESS_ARCSUF% %COVISEDIR%
SET PATH=%COVISEDIR%\%FLAG_POSTPROCESS_ARCSUF%\lib;%COVISEDIR%\%FLAG_POSTPROCESS_ARCSUF%\bin;%PATH%

ECHO ... calling %COVISEDIR%\Python\make_all_for_win32.bat
ECHO ... to generate necessary python classes for vr-prepare4 ...
%OPNEXIST% %COVISEDIR%\config\config.license.xml (
   ECHO ... WARNING: %COVISEDIR%\config\config.license.xml not found! 
   ECHO ...    generation of COVISE python classes will likely hang!
)
%OPCOPY% %INSTALLDIR%\makeBasiModIgnorelist.txt %COVISEDIR%\Python\ > NUL
CD /D %COVISEDIR%\Python
CALL %COVISEDIR%\Python\make_all_for_win32.bat

ECHO ... applying temporary vr-prepare4 fixes ...
REM @ECHO ON
%OPEXIST% %COVISEDIR%\src\visenso\branches\pyqt4\VRPUtils.py %OPDEL% %COVISEDIR%\src\visenso\branches\pyqt4\VRPUtils.py
%OPCOPY% %COVISEDIR%\src\visenso\branches\pyqt4\Utils.py %COVISEDIR%\src\visenso\branches\pyqt4\VRPUtils.py
%OPEXIST% %COVISEDIR%\src\visenso\branches\pyqt4\myauxils.py %OPDEL% %COVISEDIR%\src\visenso\branches\pyqt4\myauxils.py
%OPCOPY% %COVISEDIR%\src\visenso\branches\pyqt4\auxils.py %COVISEDIR%\src\visenso\branches\pyqt4\myauxils.py
%OPCOPY% %INSTALLDIR%\patch_coPyModules.* %COVISEDIR%\Python\ > NUL
CALL %COVISEDIR%\Python\patch_coPyModules.bat >> %COVISEDIR%\Python\coPyModules.py
REM @ECHO ON
%OPDEL% %COVISEDIR%\Python\patch_coPyModules.*
IF EXIST %INSTALLDIR%\..\Setup\install\coPyModules.py (
   REM compare generated classes against a file possibly containing more (i.e. more current?) python classes
   CALL %UNIXUTILS%\grep.exe -e "class " %INSTALLDIR%\..\Setup\install\coPyModules.py > %FILETEMP_LEFT%
   CALL %UNIXUTILS%\grep.exe -e "class " %COVISEDIR%\Python\coPyModules.py > %FILETEMP_RIGHT%
   CALL %INSTALLDIR%\..\common\compareContents.bat %FILETEMP_LEFT% %FILETEMP_RIGHT% %UNIXUTILS% > %FILETEMP_DIFF%
   CALL %UNIXUTILS%\grep.exe -c -e "" %FILETEMP_DIFF% > %FILETEMP_DIFFCNT%
   SET /P _LINECNT=<%FILETEMP_DIFFCNT%
   IF "x!_LINECNT!x" NEQ "x0x" (
      ECHO ... !_LINECNT! python classes will be patched into generated coPyModules.py:
      TYPE %FILETEMP_DIFF%
      FOR /F "delims=^" %%G IN (%FILETEMP_DIFF%) DO (
         CALL %INSTALLDIR%\..\common\cutPiece.bat "%%G" "class" %INSTALLDIR%\..\Setup\install\coPyModules.py %UNIXUTILS% >> %COVISEDIR%\Python\coPyModules.py
      )
   )
   %OPDEL% %FILETEMP_DIFF%
   %OPDEL% %FILETEMP_LEFT%
   %OPDEL% %FILETEMP_RIGHT%
   %OPDEL% %FILETEMP_DIFFCNT%
)

ECHO ... generating Qt ui classes ...
ATTRIB -R %COVISEDIR%\src\Visenso\branches\pyqt4 /S /D
REM DIR %COVISEDIR%\src\Visenso\branches\pyqt4\designerfiles /N /B *.ui > %COVISEDIR%\src\Visenso\branches\pyqt4\designerfiles\%FILETEMP_UIFILES%
REM FOR /F %%G IN (%COVISEDIR%\src\Visenso\branches\pyqt4\designerfiles\%FILETEMP_UIFILES%) DO 
FOR /F %%G IN ('DIR %COVISEDIR%\src\Visenso\branches\pyqt4\designerfiles\*.ui /N /B ') DO (
   SET _TMPPYFILE=%%G
   %OPDEL% %COVISEDIR%\src\Visenso\branches\pyqt4\!_TMPPYFILE:~0,-2!*
   CALL python.exe %COVISEDIR%\Python\bin\vr-prepare\wqtpyuic4.py -w %COVISEDIR%\src\Visenso\branches\pyqt4\designerfiles\%%G > %COVISEDIR%\src\Visenso\branches\pyqt4\%%G
)

%OPEXIST% %COVISEDIR%\src\Visenso\branches\pyqt4\designerfiles\images\StaticImages.qrc %OPPYRCC% %COVISEDIR%\src\Visenso\branches\pyqt4\designerfiles\images\StaticImages.qrc > %COVISEDIR%\src\Visenso\branches\pyqt4\StaticImages_rc.py
%OPRENAME% %COVISEDIR%\src\Visenso\branches\pyqt4\*.ui *.py
REM %OPDEL% %COVISEDIR%\src\Visenso\branches\pyqt4\designerfiles\%FILETEMP_UIFILES%
%OPEXIST% %COVISEDIR%\src\Visenso\branches\pyqt4\*.ui (
   ECHO ... for these Qt ui scripts python scripts already existed:
   DIR %COVISEDIR%\src\Visenso\branches\pyqt4\*.ui /N /B
   %OPDEL% %COVISEDIR%\src\Visenso\branches\pyqt4\*.ui
)

ECHO ... compiling Python scripts ...
REM note: only the binary vr-prepare4 is shipped; 
REM    so delete the scripts (including the ui descriptions) after compilation
CALL %INSTALLDIR%\..\common\generatePYC.bat %COVISEDIR%\Python -q
CALL %INSTALLDIR%\..\common\generatePYC.bat %COVISEDIR%\src\visenso\branches\pyqt4 -q

ECHO ... deleting Python scripts ...
CD /D %COVISEDIR%\src\visenso\branches\pyqt4
REM recursively gather a list of all subfolders
%OPEXIST% %FILETEMP_SUBDIRS% %OPDEL% %FILETEMP_SUBDIRS%
FOR /D /R %%G IN (.) DO (
   ECHO %%G>>%FILETEMP_SUBDIRS%
)
FOR /F %%G IN (%FILETEMP_SUBDIRS%) DO (
   CD /D %%G

   REM create dummy files to avoid "file not found" messages
   ECHO delete me > %FILETEMP_PYDUMMY%
   ECHO delete me > %FILETEMP_UIDUMMY%

REM   %OPDEL% *.py *.ui
   %OPEXIST% %FILETEMP_UIDUMMY% %OPDEL% %FILETEMP_UIDUMMY%
   %OPEXIST% %FILETEMP_PYDUMMY% %OPDEL% %FILETEMP_PYDUMMY%

   %OPEXIST% Makefile %OPDEL% Makefile
   %OPEXIST% PortingHints %OPDEL% PortingHints
)
%OPDEL% %COVISEDIR%\src\visenso\branches\pyqt4\%FILETEMP_SUBDIRS%
CD /D %COVISEDIR%\Python



CD /D %COVISEDIR%
ECHO ...
ECHO ... do not forget to adapt %COVISEDIR%\config\config.xml if necessary ...
ECHO ...
ECHO ...done!

GOTO END





:ERROR_ARCSUFFILEEMPTY
ECHO ...
ECHO ERROR: file %INSTALLDIR%\%FILE_ARCHSUFFIXES% does not contain any ARCHSUFFIXes!
ECHO ...
GOTO USAGE





:ERROR_NOARCHSUFFIXFILE
ECHO ...
ECHO ERROR: file %INSTALLDIR%\%FILE_ARCHSUFFIXES% not found!
ECHO ...
GOTO USAGE





:USAGE
ECHO ...
ECHO deploys a COVISE shipment
ECHO ...
ECHO usage:
ECHO %0 
ECHO    [COVISEDIR] <- install path of covise sources (e. g. D:\TRUNK\covise)
ECHO    [destination path of shipment]
ECHO    [path to custom license files]
ECHO    [path to unix utils, like grep, sed and head]
ECHO    [folder name containing files specific to distribution, if not supplied
ECHO        then ARCHSUFFIX is assumed as folder name]
ECHO ...
ECHO note: the ARCHSUFFIXes to be shipped are to be listed one line each in
ECHO    the file %INSTALLDIR%\%FILE_ARCHSUFFIXES%;
ECHO    this script expects the EXTERNLIBS environment variable to be setup
ECHO    in advance;
ECHO ...
ECHO called batch scripts:
ECHO     common.VISENSO.bat
ECHO     make_all_for_win32.bat
ECHO     patch_coPyModules.bat
ECHO     pyuic4.bat
ECHO     generatePYC.bat
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1





:END
ENDLOCAL
EXIT /B 0