IF /i "%ARCHSUFFIX%" == "win32opt" (
  set USE_OPT_LIBS=1
) ELSE (
  IF /i "%ARCHSUFFIX%" == "vistaopt" (
    set USE_OPT_LIBS=1
  ) ELSE (
    IF /i "%ARCHSUFFIX%" == "vistampiopt" (
      set USE_OPT_LIBS=1
    ) ELSE (
      IF /i "%ARCHSUFFIX%" == "amdwin64opt" (
        set USE_OPT_LIBS=1
      ) ELSE (
        IF /i "%ARCHSUFFIX%" == "zackelopt" (
          set USE_OPT_LIBS=1
        ) ELSE (
          IF /i "%ARCHSUFFIX%" == "angusopt" (
            set USE_OPT_LIBS=1
          ) ELSE (
            IF /i "%ARCHSUFFIX%" == "yorooopt" (
              set USE_OPT_LIBS=1
            ) ELSE (
              IF /i "%ARCHSUFFIX%" == "berrendaopt" (
                set USE_OPT_LIBS=1
              ) ELSE (
                IF /i "%ARCHSUFFIX%" == "tamarauopt" (
                  set USE_OPT_LIBS=1
                ) ELSE ( 
                  IF /i "%ARCHSUFFIX%" == "zebuopt" (
                    set USE_OPT_LIBS=1
                  ) ELSE (
                    set USE_OPT_LIBS=0
                    echo DEBUG-Build !!! 
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)


rem   start microsoft development environment
rem   =======================================
rem
rem If VS2003 or VS2005 was installed in a non-standard location you have to set VCVARS32 !
rem 

set PROGFILES=%ProgramFiles%
if defined ProgramFiles(x86)  set PROGFILES=%ProgramFiles(x86)%
rem echo  %VS100COMNTOOLS%
cd


if "%ARCHSUFFIX:~0,5%" EQU "win32" (
    call "%PROGFILES%"\"Microsoft Visual Studio .NET 2003"\Vc7\bin\vcvars32.bat
) else if "%ARCHSUFFIX:~0,5%" EQU "vista" (
    call "%VS80COMNTOOLS%"\..\..\Vc\bin\vcvars32.bat"
) else if "%ARCHSUFFIX:~0,6%" EQU "zackel" (
    cd /d "%VS90COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x86
    cd /d "%COVISEDIR%"\
) else if "%ARCHSUFFIX:~0,5%" EQU "yoroo" (
    cd /d "%VS100COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x86
    cd /d "%COVISEDIR%"\
) else if "%ARCHSUFFIX:~0,7%" EQU "tamarau" (
    cd /d "%VS110COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x64
    cd /d "%COVISEDIR%"\
) else if "%ARCHSUFFIX:~0,4%" EQU "zebu" (

    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsDevCmd.bat" call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
    if defined VS150COMNTOOLS% (
    
	)else (
    cd /d "%VS140COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x64
    cd /d "%COVISEDIR%"\
	)
) else if "%ARCHSUFFIX:~0,8%" EQU "berrenda" (
if defined VS110COMNTOOLS  (
    cd /d "%VS110COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x64
) else (
    cd /d "%VS100COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x64
	)
    cd /d "%COVISEDIR%"\
) else if "%ARCHSUFFIX:~0,5%" EQU "angus" (
    cd /d "%VS90COMNTOOLS%"\..\..\vc
    call vcvarsall.bat x64
    cd /d "%COVISEDIR%"\
) else if "%ARCHSUFFIX:~0,8%" EQU "amdwin64"   (
    cd /d "%VS80COMNTOOLS%"\..\..\vc
    call vcvarsall.bat x64
    cd /d "%COVISEDIR%"\
) else if defined VCVARS32 (
    call "%VCVARS32%"
)



if not defined SWIG_HOME  (
   set "SWIG_HOME=%EXTERNLIBS%\swig"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\swig"
)

if not defined BISON_HOME  (
   set "BISON_HOME=%EXTERNLIBS%\bison"
   set "PATHPRE=%PATHPRE%;%EXTERNLIBS%\bison\bin"
)

if not defined FLEX_HOME  (
   set "FLEX_HOME=%EXTERNLIBS%\flex"
   set "PATHPRE=%PATHPRE%;%EXTERNLIBS%\flex\bin"
)

if not defined FREETYPE_DIR  (
   set "FREETYPE_DIR=%EXTERNLIBS%\freetype"
)

if not defined THREEDTK_HOME  (
   set "THREEDTK_HOME=%EXTERNLIBS%\3dtk"
   set "PATHPRE=%PATHPRE%;%EXTERNLIBS%\3dtk\bin"
)

if not defined BOOST_HOME  (
   set "BOOST_HOME=%EXTERNLIBS%\boost"
   set "BOOST_ROOT=%EXTERNLIBS%\boost"
   set "PATHPRE=%PATHPRE%;%EXTERNLIBS%\boost\lib"
   set "BOOST_INCLUDEDIR=%EXTERNLIBS%\boost\include"
)
set "SWIG=%SWIG_HOME%\swig"
set "SWIG_INCLUDES=-I%SWIG_HOME%\Lib -I%SWIG_HOME%\Lib\python -I%SWIG_HOME%\Lib\typemaps"

if not defined MPI_HOME  (
 set "MPI_HOME=%EXTERNLIBS%\msmpi\lib"
 set "MPI_INCPATH=%EXTERNLIBS%\msmpi\include"
 set "PATHADD=%PATHADD%;%EXTERNLIBS%\msmpi\bin"
)
REM if not defined MPI_HOME  (
REM  set "MPI_HOME=%EXTERNLIBS%\MPICH2\lib"
REM  set "MPI_INCPATH=%EXTERNLIBS%\MPICH2\include"
REM  set "PATHADD=%PATHADD%;%EXTERNLIBS%\MPICH2\bin"
REM  IF /i "%ARCHSUFFIX%" == "vistampi" (
REM     set "MPI_LIBS=-L%EXTERNLIBS%\MPICH2\lib -lcxx -lmpi"
REM     set MPI_DEFINES=HAS_MPI
REM  )
REM  IF /i "%ARCHSUFFIX%" == "vistampiopt" (
REM     set "MPI_LIBS=-L%EXTERNLIBS%\MPICH2\lib -lcxxd -lmpid"
REM     set MPI_DEFINES=HAS_MPI
REM  )
REM  IF /i "%ARCHSUFFIX%" == "angusmpi" (
REM     set "MPI_LIBS=-L%EXTERNLIBS%\MPICH2\lib -lcxx -lmpi"
REM     set MPI_DEFINES=HAS_MPI
REM  )
REM  IF /i "%ARCHSUFFIX%" == "angusmpiopt" (
REM     set "MPI_LIBS=-L%EXTERNLIBS%\MPICH2\lib -lcxx -lmpi"
REM     set MPI_DEFINES=HAS_MPI
REM  )
REM )

if not defined OPENSSL_HOME  (
   set "OPENSSL_HOME=%EXTERNLIBS%\OpenSSL"
   set "OPENSSL_INCPATH=%EXTERNLIBS%\OpenSSL\include"
   set "OPENSSL_DEFINES=HAVE_SSL"
   set "PATHPRE=%PATHPRE%;%EXTERNLIBS%\OpenSSL\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "OPENSSL_LIBS=-L%EXTERNLIBS%/OpenSSL/lib -llibeay32 -lssleay32"
   ) else (
      set "OPENSSL_LIBS=-L%EXTERNLIBS%/OpenSSL/lib -llibeay32d -lssleay32d"
   )
)

if not defined GDCM_HOME  (
   if exist %EXTERNLIBS%\GDCM\lib\gdcmCommon.lib (
     set "GDCM_HOME=%EXTERNLIBS%\GDCM"
     set "GDCM_INCPATH=%EXTERNLIBS%\GDCM\include"
     set "GDCM_RESOURCE_PATH=%EXTERNLIBS%\GDCM\etc"
     set "GDCM_DEFINES=HAVE_GDCM"
     if "%USE_OPT_LIBS%" == "1" (
        set "PATHADD=%PATHADD%;%EXTERNLIBS%\GDCM\bin\Release"
        set "GDCM_LIBS=-L%EXTERNLIBS%/GDCM/lib -lgdcmcharls -lgdcmCommon -lgdcmconv -lgdcmDICT -lgdcmDSED -lgdcmexpat -lgdcmgetopt -lgdcmimg -lgdcminfo -lgdcmIOD -lgdcmjpeg8 -lgdcmjpeg12 -lgdcmjpeg16 -lgdcmMSFF -lgdcmopenjpeg -lgdcmraw -lgdcmtar -lgdcmzlib"
     ) else (
        set "PATHADD=%PATHADD%;%EXTERNLIBS%\GDCM\bin\Debug"
        set "GDCM_LIBS=-L%EXTERNLIBS%/GDCM/lib -lgdcmcharlsD -lgdcmCommonD -lgdcmconvD -lgdcmDICTD -lgdcmDSEDD -lgdcmexpatD -lgdcmgetoptD -lgdcmimgD -lgdcminfoD -lgdcmIODD -lgdcmjpeg8D -lgdcmjpeg12D -lgdcmjpeg16D -lgdcmMSFFD -lgdcmopenjpegD -lgdcmrawD -lgdcmtarD -lgdcmzlibD"
     )
   )
)

if not defined GSOAP_HOME (
   set "GSOAP_HOME=%EXTERNLIBS%\gsoap\gsoap"
   set "GSOAP_IMPORTDIR=%EXTERNLIBS%\gsoap\gsoap"
   set "GSOAP_INCPATH=%EXTERNLIBS%\gsoap\gsoap\include"
   set "GSOAP_DEFINES=HAVE_GSOAP"
   set "GSOAP_BINDIR=%EXTERNLIBS%\gsoap\gsoap\bin\win32"
)

if /I "%ARCHSUFFIX%" == "vistaopt" (
   set HAVE_CUDA=1
) else if /I "%ARCHSUFFIX%" == "vista" (
   set HAVE_CUDA=1
) else if /I "%ARCHSUFFIX%" == "angus" (
   set HAVE_CUDA=1
) else if /I "%ARCHSUFFIX%" == "angusopt" (
   set HAVE_CUDA=1
)

   set HAVE_CUDA=1
if defined HAVE_CUDA (
if not defined CUDA_HOME  (
   if defined CUDA_BIN_PATHDONTBECAUSEOFSPACES  (
      REM set "CUDA_HOME=%CUDA_INC_PATH%\.."
      REM set "CUDA_INCPATH=%CUDA_INC_PATH%"
      set "CUDA_HOME=%EXTERNLIBS%\Cuda"
      set "CUDA_BIN_PATH=%EXTERNLIBS%\Cuda\bin"
      set "CUDA_INCPATH=%EXTERNLIBS%\Cuda\include"
      set "CUDA_DEFINES=HAVE_CUDA"
      set "CUDA_SDK_HOME=%EXTERNLIBS%\CUDA"
      set "CUDA_SDK_INCPATH=%EXTERNLIBS%\CUDA\include %EXTERNLIBS%\CUDA\common\inc"
      set "PATHADD=%PATHADD%;%CUDA_BIN_PATH%"
      if "%USE_OPT_LIBS%" == "1" (
	     set "PATHADD=%PATHADD%;%EXTERNLIBS%\Cuda\bin\win32\Release;%EXTERNLIBS%\cudpp\bin"
         set "CUDA_LIBS=-L%CUDA_LIB_PATH% -L%EXTERNLIBS%\Cuda\common\lib -lcuda -lcudart -lcutil32"
      ) else (
	     set "PATHADD=%PATHADD%;%EXTERNLIBS%\Cuda\bin\win32\Debug;%EXTERNLIBS%\cudpp\bin"
         set "CUDA_LIBS=-L%CUDA_LIB_PATH% -L%EXTERNLIBS%\Cuda\common\lib -lcuda -lcudart -lcutil32D"
      )
   ) else if exist %EXTERNLIBS%\Cuda (
      set "CUDA_HOME=%EXTERNLIBS%\Cuda"
      set "CUDA_BIN_PATH=%EXTERNLIBS%\Cuda\bin"
      set "CUDA_INCPATH=%EXTERNLIBS%\Cuda\include"
      set "CUDA_DEFINES=HAVE_CUDA"
      set "CUDA_SDK_HOME=%EXTERNLIBS%\CUDA"
      set "CUDA_SDK_INCPATH=%EXTERNLIBS%\CUDA\include %EXTERNLIBS%\CUDA\common\inc"
      set "PATHADD=%PATHADD%;%CUDA_BIN_PATH%"
      if "%USE_OPT_LIBS%" == "1" (
	     set "PATHADD=%PATHADD%;%EXTERNLIBS%\Cuda\bin\win32\Release"
         set "CUDA_LIBS=-L%EXTERNLIBS%\Cuda\lib -L%EXTERNLIBS%\Cuda\common\lib -lcuda -lcudart -lcutil64"
      ) else (
	     set "PATHADD=%PATHADD%;%EXTERNLIBS%\CUDA\bin\win32\Debug"
         set "CUDA_LIBS=-L%EXTERNLIBS%\Cuda\lib -L%EXTERNLIBS%\Cuda\common\lib -lcuda -lcudart -lcutil64D"
      )
   )
)
)

if not defined CUDPP_HOME  (
   set "CUDPP_HOME=%EXTERNLIBS%\cudpp"
   set "CUDPP_INCPATH=%EXTERNLIBS%\cudpp\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\cudpp\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "CUDPP_LIBS=-L%EXTERNLIBS%\cudpp\lib -lcudpp64"
   ) else (
      set "CUDPP_LIBS=-L%EXTERNLIBS%\cudpp\lib -lcudpp64d"
   )
)

if not defined FFMPEG_HOME  (
   set "FFMPEG_HOME=%EXTERNLIBS%\ffmpeg"
   set "FFMPEG_DEFINES=VV_FFMPEG"
   set "FFMPEG_INCPATH=%EXTERNLIBS%\ffmpeg\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\ffmpeg\bin"
   set "FFMPEG_LIBS=-L%EXTERNLIBS%\ffmpeg\lib -lavutil -lavcodec -lswscale"
)

if not defined HDF5_ROOT  (
   set "HDF5_ROOT=%EXTERNLIBS%\hdf5"
)

if not defined PCL_HOME  (
   set "PCL_HOME=%EXTERNLIBS%\pcl"
   set "PCL_INCPATH=%EXTERNLIBS%\pcl\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\pcl\bin"
)

if not defined OPENCV_HOME  (
   set "OPENCV_HOME=%EXTERNLIBS%\opencv"
   set "OPENCV_ROOT=%EXTERNLIBS%\opencv"
    set "OPENCV_DEFINES=VV_FFMPEG"
   set "OPENCV_INCPATH=%EXTERNLIBS%\opencv\build\include"
    IF /i "%ARCHSUFFIX%" == "tamarau" (
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\opencv\build\x64\vc11\bin"
    ) ELSE (
    IF /i "%ARCHSUFFIX%" == "tamarauopt" (
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\opencv\build\x64\vc11\bin"
    ) 
 else if "%ARCHSUFFIX:~0,4%" EQU "zebu" (
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\opencv\build\x64\vc14\bin"
)
ELSE (
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\opencv\build\x64\vc10\bin"
   )
   )
)
if not defined OSCPACK  (
   set "OSCPACK_HOME=%EXTERNLIBS%\oscpack"
   set "OSCPACK_DEFINES=HAVE_OSCPACK __x86_64__"
   set "OSCPACK_INCPATH=%EXTERNLIBS%\oscpack\include"
   if "%USE_OPT_LIBS%" == "1" (
     set "OSCPACK_LIBS=-L%EXTERNLIBS%\oscpack\lib -loscpack -lWs2_32 -lWinmm"
   ) else (
     set "OSCPACK_LIBS=-L%EXTERNLIBS%\oscpack\lib -loscpackd -lWs2_32 -lWinmm"
   )
)

if not defined BULLET_HOME  (
   set "BULLET_HOME=%EXTERNLIBS%\bullet"
   set "BULLET_DEFINES=HAVE_BULLET"
   set "BULLET_INCPATH=%EXTERNLIBS%\bullet\include"
   
   if "%USE_OPT_LIBS%" == "1" (
      set "BULLET_LIBS=-L%EXTERNLIBS%\bullet\lib -lBulletCollision -lBulletDynamics -lGIMPACTUtils -lLinearMath"
      set "OSGBULLET_LIBS=-L%EXTERNLIBS%\OpenSceneGraph\lib -losgbBullet"
   ) else (
      set "BULLET_LIBS=-L%EXTERNLIBS%\bullet\lib -lBulletCollisiond -lBulletDynamicsd -lGIMPACTUtilsd -lLinearMathd"
      set "OSGBULLET_LIBS=-L%EXTERNLIBS%\OpenSceneGraph\lib -losgbBulletd"
   )
)

if not defined ICAL_HOME  (
   set "ICAL_HOME=%EXTERNLIBS%\libical"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\libical\bin"
)
if not defined ALVAR_HOME  (
   set "ALVAR_HOME=%EXTERNLIBS%\ALVAR"
   set "ALVAR_DEFINES=HAVE_ALVAR"
   set "ALVAR_INCPATH=%EXTERNLIBS%\ALVAR\include"
   set "ALVAR_PLUGIN_PATH=%EXTERNLIBS%\ALVAR\bin\alvarplugins"
   set "ALVAR_LIBRARY_PATH=%EXTERNLIBS%\ALVAR\bin"
   
   
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\ALVAR\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "ALVAR_LIBS=-L%EXTERNLIBS%\ALVAR\lib -lALVARCollision -lALVARDynamics -lGIMPACTUtils -lLinearMath"
      set "OSGALVAR_LIBS=-L%EXTERNLIBS%\OpenSceneGraph\lib -losgbALVAR"
   ) else (
      set "ALVAR_LIBS=-L%EXTERNLIBS%\ALVAR\lib -lALVARCollisiond -lALVARDynamicsd -lGIMPACTUtilsd -lLinearMathd"
      set "OSGALVAR_LIBS=-L%EXTERNLIBS%\OpenSceneGraph\lib -losgbALVARd"
   )
)

if not defined GLEW_HOME  (
   set "GLEW_HOME=%EXTERNLIBS%\glew"
   set "GLEW_INCPATH=%EXTERNLIBS%\glew\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\glew\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "GLEW_LIBS=-L%EXTERNLIBS%\glew\lib -lglew32"
   ) else (
      set "GLEW_LIBS=-L%EXTERNLIBS%\glew\lib -lglew32"
   )
)

if not defined FMOD_HOME  (
   set "FMOD_HOME=%EXTERNLIBS%\fmod"
   set "FMOD_INCPATH=%EXTERNLIBS%\fmod\inc"
   set "FMOD_DEFINES=HAVE_FMOD"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\fmod\bin"
   
   if "%ARCHSUFFIX:~0,5%" EQU "angus" (
      if "%USE_OPT_LIBS%" == "1" (
         set "FMOD_LIBS=-L%EXTERNLIBS%\fmod\lib -lfmodex64_vc -lfmod_event64"
      ) else (
         set "FMOD_LIBS=-L%EXTERNLIBS%\fmod\lib -lfmodexL64_vc -lfmod_event64L"
      )
   ) else (
      if "%USE_OPT_LIBS%" == "1" (
         set "FMOD_LIBS=-L%EXTERNLIBS%\fmod\lib -lfmodex_vc -lfmod_event"
      ) else (
         set "FMOD_LIBS=-L%EXTERNLIBS%\fmod\lib -lfmodexL_vc -lfmod_eventL"
      )
   )
)
if not defined ABAQUS_HOME  (
   set "ABAQUS_HOME=%EXTERNLIBS%\Abaqus"
   set "ABAQUS_INCPATH=%EXTERNLIBS%\Abaqus\include"
   set "ABAQUS_DEFINES=HKS_NT HAVE_ABAQUS"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\Abaqus\bin"
   set "ABAQUS_LIBS=-L%EXTERNLIBS%\Abaqus\lib -lstandardB -lstandardU -lABQDDB_Core_import -lABQUTI_CoreUtils_import -lABQUTI_BasicUtils_import -lABQDDB_Odb_import -lABQDDB_ODB_API_import"
)

if not defined GAALET_HOME  (
   set "GAALET_HOME=%EXTERNLIBS%\gaalet"
   set "GAALET_INCPATH=%EXTERNLIBS%\gaalet\include"
   set "GAALET_DEFINES=HAVE_GAALET"
   set "GAALET_LIBS="
)

if not defined FFTW3_HOME  (
   set "FFTW3_HOME=%EXTERNLIBS%\fftw"
   set "FFTW3_INCPATH=%EXTERNLIBS%\fftw\include"
   set "FFTW3_DEFINES=HAVE_FFTW3"
   set "FFTW3_LIBS=-L%EXTERNLIBS%\fftw\lib -llibfftw3-3"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\fftw\bin"
)


if not defined COLLADA_HOME  (
   set "COLLADA_HOME=%EXTERNLIBS%\collada"
   set "COLLADA_INCPATH=%EXTERNLIBS%\collada\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\collada\lib"
   if "%USE_OPT_LIBS%" == "1" (
      set "COLLADA_LIBS=-L%EXTERNLIBS%\collada\lib -llibcollada14dom21.lib"
   ) else (
      set "COLLADA_LIBS=-L%EXTERNLIBS%\collada\lib -llibcollada14dom21-d.lib"
   )
)


if not defined TOUCHLIB_HOME  (
   set "TOUCHLIB_HOME=%EXTERNLIBS%\touchLib"
   set "TOUCHLIB_INCPATH=%EXTERNLIBS%\touchLib\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\touchLib\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "TOUCHLIB_LIBS=-L%EXTERNLIBS%\touchLib\lib -ltouchlib"
   ) else (
      set "TOUCHLIB_LIBS=-L%EXTERNLIBS%\touchLib\lib -ltouchlibd"
   )
)


if not defined OSGEPHEMERIS_HOME  (
   set "OSGEPHEMERIS_HOME=%EXTERNLIBS%\osgEphemeris"
   set "OSGEPHEMERIS_INCPATH=%EXTERNLIBS%\osgEphemeris\include"
   set "OSGEPHEMERIS_DEFINES=HAVE_OSGEPHEMERIS"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\osgEphemeris\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "OSGEPHEMERIS_LIBS=-L%EXTERNLIBS%\osgEphemeris\lib -losgEphemeris"
   ) else (
      set "OSGEPHEMERIS_LIBS=-L%EXTERNLIBS%\osgEphemeris\lib -losgEphemerisd"
   )
)

if not defined OSGCAL_HOME  (
   set "OSGCAL_HOME=%EXTERNLIBS%\osgCal"
   set "OSGCAL_INCPATH=%EXTERNLIBS%\osgCal\include"
   set "OSGCAL_DEFINES=HAVE_OSGCAL"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\osgCal\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "OSGCAL_LIBS=-L%EXTERNLIBS%\osgCal\lib -losgCal"
   ) else (
      set "OSGCAL_LIBS=-L%EXTERNLIBS%\osgCal\lib -losgCald"
   )
)

if not defined  OPENSCENEGRAPH_HOME (
   if exist %EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe (
     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --version-number') do @set OSG_VER_NUM=%%v
     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --so-number') do @set OSG_SO_VER=%%v
     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --openthreads-soversion-number') do @set OSG_OT_SO_VER=%%v
   )

   set "OPENSCENEGRAPH_HOME=%EXTERNLIBS%\OpenSceneGraph"
   set "OSG_DIR=%EXTERNLIBS%\OpenSceneGraph"
   set "OPENSCENEGRAPH_INCPATH=%EXTERNLIBS%\OpenSceneGraph\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\OpenSceneGraph\bin;"
   if "%USE_OPT_LIBS%" == "1" (
      set "OPENSCENEGRAPH_LIBS=%OSGNV_LIBS% -L%EXTERNLIBS%\OpenSceneGraph\lib -losg -losgDB -losgUtil -losgViewer -losgParticle -losgText -losgSim -losgGA -losgFX -lOpenThreads"
   ) else (
      set "OPENSCENEGRAPH_LIBS=%OSGNV_LIBS% -L%EXTERNLIBS%\OpenSceneGraph\lib -losgD -losgDBd -losgUtilD -losgViewerD -losgParticleD -losgTextD -losgSimD -losgGAd -losgFXd -lOpenThreadsD"
   )
)

if not defined  OPENTHREADS_HOME (
   if exist %EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe (
     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --version-number') do @set OSG_VER_NUM=%%v
     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --so-number') do @set OSG_SO_VER=%%v
     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --openthreads-soversion-number') do @set OSG_OT_SO_VER=%%v
   )

   set "OPENTHREADS_HOME=%EXTERNLIBS%\OpenSceneGraph"
   set "OPENTHREADS_INCPATH=%EXTERNLIBS%\OpenSceneGraph\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\OpenSceneGraph\bin;"
   if "%USE_OPT_LIBS%" == "1" (
      set "OPENTHREADS_LIBS=-L%EXTERNLIBS%\OpenSceneGraph\lib -lOpenThreads"
   ) else (
      set "OPENTHREADS_LIBS=-L%EXTERNLIBS%\OpenSceneGraph\lib -lOpenThreadsD"
   )
)

if "%OSG_VER_NUM%" NEQ "" (
  if not defined OSG_LIBRARY_PATH (
    set "OSG_LIBRARY_PATH=%EXTERNLIBS%\OpenSceneGraph\bin\osgPlugins-%OSG_VER_NUM%"
  )
)


if not defined DXSDK_HOME (
   IF not defined DXSDK_DIR (
     REM DXSDK_DIR is not set ! Try in EXTERNLIBS?
     set "DXSDK_HOME=%EXTERNLIBS%\dxsdk"
     set "DXSDK_INCPATH=%EXTERNLIBS%\dxsdk\include"
     IF /I "%ARCHSUFFIX%" == "amdwin64opt" (
        set "DXSDK_LIBS=-L%EXTERNLIBS%\dxsdk\lib\x64 -ldxguid -ldxerr9 -ldinput8 -lcomctl32"
        goto DX_LIB_PATH_SET
     )
     IF /I "%ARCHSUFFIX%" == "amdwin64" (
        set "DXSDK_LIBS=-L%EXTERNLIBS%\dxsdk\lib\x64 -ldxguid -ldxerr9 -ldinput8 -lcomctl32"
        goto DX_LIB_PATH_SET
     )
     set "DXSDK_LIBS=-L%EXTERNLIBS%\dxsdk\lib -L%EXTERNLIBS%\dxsdk\lib\x86 -ldxguid -ldxerr9 -ldinput8 -lcomctl32"
   ) ELSE (
     REM DXSDK_DIR is set so use it
     echo DXSDK_DIR is set to "%DXSDK_DIR%"
     set "DXSDK_HOME=%DXSDK_DIR%"
     set "DXSDK_INCPATH=%DXSDK_DIR%\include"
	 
     if exist "%DXSDK_DIR%lib\x64\dxerr9.lib" (
	 set ERRLIB=-ldxerr9
     )	 
     IF /I "%ARCHSUFFIX%" == "amdwin64opt" (
        set "DXSDK_LIBS=-L%DXSDK_DIR%lib\x64 -ldxguid %ERRLIB% -ldinput8 -lcomctl32"
        goto DX_LIB_PATH_SET
     )
     IF /I "%ARCHSUFFIX%" == "amdwin64" (
        set "DXSDK_LIBS=-L%DXSDK_DIR%lib\x64 -ldxguid %ERRLIB% -ldinput8 -lcomctl32"
        goto DX_LIB_PATH_SET
     )
     IF /I "%ARCHSUFFIX%" == "angusopt" (
        set "DXSDK_LIBS=-L%DXSDK_DIR%lib\x64 -ldxguid %ERRLIB% -ldinput8 -lcomctl32"
        goto DX_LIB_PATH_SET
     )
     IF /I "%ARCHSUFFIX%" == "angus" (
        set "DXSDK_LIBS=-L%DXSDK_DIR%lib\x64 -ldxguid %ERRLIB% -ldinput8 -lcomctl32"
        goto DX_LIB_PATH_SET
     )
     set "DXSDK_LIBS=-L%DXSDK_DIR%lib -L%DXSDK_DIR%lib\x86 -ldxguid -ldxerr9 -ldinput8 -lcomctl32"
   )
)
:DX_LIB_PATH_SET


if not defined OSG_HOME (
   set "OSG_HOME=%EXTERNLIBS%\OpenSG"
   set "OSG_INCPATH=%EXTERNLIBS%\OpenSG\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\OpenSG\lib;%EXTERNLIBS%\tiff\bin"

   if "%USE_OPT_LIBS%" == "1" (
      set "OSG_LIBS=-L%EXTERNLIBS%\OpenSG\lib OSGBase.lib OSGSystem.lib OSGWindowWIN32.lib"
   ) else (
      set "OSG_LIBS=-L%EXTERNLIBS%\OpenSG\lib OSGBaseD.lib  OSGSystemD.lib OSGWindowWIN32D.lib"
   )
)


if not defined DSIO_HOME (
   set "DSIO_HOME=%EXTERNLIBS%\dsio"
   set "DSIO_INCPATH=%EXTERNLIBS%\dsio\include"
   set "DSIO_DEFINES=HAVE_DSIO"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\dsio\lib"
   set "DSIO_LIBS=-L%EXTERNLIBS%\dsio\lib dsio_md.lib"
)

if not defined DSVL_HOME (
   set "DSVL_HOME=%EXTERNLIBS%\dsvl"
)


if not defined CG_HOME ( 
   set "CG_HOME=%EXTERNLIBS%\Cg"
   set "CG_INCPATH=%EXTERNLIBS%\Cg\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\Cg\bin"
   set "CG_LIBS=-L%EXTERNLIBS%\Cg\lib -lcg -lcgGL"
   set CG_DEFINES=HAVE_CG
)

if not defined PCAN_HOME ( 
   if exist %EXTERNLIBS%\PCAN-Light\nul (
    set "PCAN_HOME=%EXTERNLIBS%\PCAN-Light"
REM    set "PATHADD=%PATHADD%;%EXTERNLIBS%\PCAN-Light"
    set "PCAN_INCPATH=%EXTERNLIBS%\PCAN-Light\Include"
    set "PCAN_LIBS=-L%EXTERNLIBS%\PCAN-Light\Lib\VC_LIB -lPcan_pci"
    set PCAN_DEFINES=HAVE_PCAN
  )
)



if not defined JT_HOME  ( 
   set "JT_HOME=%EXTERNLIBS%\JT"
   set "JT_INCPATH=%EXTERNLIBS%\JT\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\JT\lib"
   set "JT_LIBS=-L%EXTERNLIBS%\JT\lib JtTk43.lib"
   set JT_DEFINES=HAVE_JT
)



if not defined QT_HOME ( 
   REM QT_HOME is not set... check QTDIR
   IF not defined QTDIR (
     REM QTDIR is not set ! Try in EXTERNLIBS
     set "QT_HOME=%EXTERNLIBS%\qt5"
     set "QT_SHAREDHOME=%EXTERNLIBS%\qt5"
     set "QTDIR=%EXTERNLIBS%\qt5"
     set "QT_INCPATH=%EXTERNLIBS%\qt5\include"
     set "QT_LIBPATH=%EXTERNLIBS%\qt5\lib"
	 set "PATH=%EXTERNLIBS%\qt5\bin;%EXTERNLIBS%\qt5\lib;%PATH%"
	 set "QT_QPA_PLATFORM_PLUGIN_PATH=%EXTERNLIBS%\qt5\plugins\platforms"   rem tested for qt5 on win7, visual studio 2010
   ) ELSE (
     REM QTDIR is set so try to use it !
     REM Do a simple sanity-check...
     IF NOT EXIST "%QTDIR%\.qmake.cache" (
       echo *** WARNING: .qmake.cache NOT found !
       echo ***          Check QTDIR or simply do NOT set QT_HOME and QTDIR to use the version from EXTERNLIBS!
       pause
     )
     REM Set QT_HOME according to QTDIR. If User ignores any warnings before he will find himself in a world of pain! 
     set "QT_HOME=%QTDIR%"
     set "QT_SHAREDHOME=%QTDIR%"
     set "QT_INCPATH=%QTDIR%\include"
     set "QT_LIBPATH=%QTDIR%\lib"
	 set "PATH=%QTDIR%\bin;%QTDIR%\lib;%PATH%"
	 set "QT_QPA_PLATFORM_PLUGIN_PATH=%QTDIR%\plugins\platforms"  
   )
)

  
if not defined PROJ4_HOME  (
   set "PROJ4_HOME=%EXTERNLIBS%\Proj4"
   set "PROJ4_INCPATH=%EXTERNLIBS%\Proj4\include"
   set "PROJ_LIB=%EXTERNLIBS%\Proj4\nad"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\Proj4\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "PROJ4_LIBS=-L%EXTERNLIBS%\Proj4\lib -lproj4"
   ) else (
      set "PROJ4_LIBS=-L%EXTERNLIBS%\Proj4\lib -lproj4D"
   )
)
if not defined ZLIB_HOME  (
   set "ZLIB_HOME=%EXTERNLIBS%\zlib"
   set "ZLIB_INCPATH=%EXTERNLIBS%\zlib\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\zlib\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "ZLIB_LIBS=-L%EXTERNLIBS%\zlib\lib -lzlib1_i"
   ) else (
      set "ZLIB_LIBS=-L%EXTERNLIBS%\zlib\lib -lzlib1D_i"
   )
)



if not defined JPEG_HOME  (
   set "JPEG_HOME=%EXTERNLIBS%\jpeg"
   set "JPEG_INCPATH=%EXTERNLIBS%\jpeg\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\jpeg\lib"
   if "%USE_OPT_LIBS%" == "1" (
      set "JPEG_LIBS=-L%EXTERNLIBS%\jpeg\lib -ljpeg_i"
   ) else (
      set "JPEG_LIBS=-L%EXTERNLIBS%\jpeg\lib -ljpegD_i"
   )
)


if not defined FREEGLUT_HOME  (
   set "FREEGLUT_HOME=%EXTERNLIBS%\freeglut"
   set "FREEGLUT_INCPATH=%EXTERNLIBS%\freeglut\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\freeglut\lib"
   if "%USE_OPT_LIBS%" == "1" (
      set "FREEGLUT_LIBS=-L%EXTERNLIBS%\freeglut\lib -lfreeglut_static"
   ) else (
      set "FREEGLUT_LIBS=-L%EXTERNLIBS%\freeglut\lib -lfreeglutD_static"
   )
)


if not defined PNG_HOME (
   set "PNG_HOME=%EXTERNLIBS%\png"
   set "PNG_INCPATH=%EXTERNLIBS%\png\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\png\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "PNG_LIBS=-L%EXTERNLIBS%\png\lib -llibpng13_i"
   ) else (
      set "PNG_LIBS=-L%EXTERNLIBS%\png\lib -llibpng13D_i"
   )
)


if not defined OPENCRG_HOME (
   set "OPENCRG_HOME=%EXTERNLIBS%\OpenCRG"
   set "OPENCRG_INCPATH=%EXTERNLIBS%\OpenCRG\include"
   if "%USE_OPT_LIBS%" == "1" (
      set "OPENCRG_LIBS=-L%EXTERNLIBS%\OpenCRG\lib -lOpenCRG"
   ) else (
      set "OPENCRG_LIBS=-L%EXTERNLIBS%\OpenCRG\lib -lOpenCRGD"
   )
)


if not defined TIFF_HOME (
   set "TIFF_HOME=%EXTERNLIBS%\tiff"
   set "TIFF_INCPATH=%EXTERNLIBS%\tiff\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\tiff\lib"
   if "%USE_OPT_LIBS%" == "1" (
      set "TIFF_LIBS=-L%EXTERNLIBS%\tiff\lib -ltiff_i"
   ) else (
      set "TIFF_LIBS=-L%EXTERNLIBS%\tiff\lib -ltiffD_i"
   )
)



if not defined AUDIOFILE_HOME (
   set "AUDIOFILE_HOME=%EXTERNLIBS%\audiofile"
   set "AUDIOFILE_INCPATH=%EXTERNLIBS%\audiofile\include"
   set AUDIOFILE_DEFINES=HAVE_AUDIOFILE
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\audiofile\lib"
   if "%USE_OPT_LIBS%" == "1" (
     set "AUDIOFILE_LIBS=-L%EXTERNLIBS%\audiofile\lib -laudiofile_i"
   ) else (
     set "AUDIOFILE_LIBS=-L%EXTERNLIBS%\audiofile\lib -laudiofileD_i"
   )
)


if not defined OPENAL_HOME (
   set "OPENAL_HOME=%EXTERNLIBS%\openal"
   set "OPENAL_INCPATH=%EXTERNLIBS%\openal\include"
   set OPENAL_DEFINES=HAVE_OPENAL
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\openal\lib"
   if "%USE_OPT_LIBS%" == "1" (
     set "OPENAL_LIBS=-L%EXTERNLIBS%\openal\lib -lOpenAL32 -lalut"
   ) else (
     set "OPENAL_LIBS=-L%EXTERNLIBS%\openal\lib -lOpenAL32D -lalutD"
   )
)



if not defined SDL_HOME (
      set "SDL_HOME=%EXTERNLIBS%\sdl"
      set "SDL_INCPATH=%EXTERNLIBS%\sdl\include"
      set SDL_DEFINES=HAVE_SDL
      set "PATHADD=%PATHADD%;%EXTERNLIBS%\sdl\bin"
      set "SDL_LIBS=-L%EXTERNLIBS%\sdl\lib -lsdl"
)



if not defined COIN3D_HOME  (
   set "COIN3D_HOME=%EXTERNLIBS%\Coin3d"
   set "COIN3D_INCPATH=%EXTERNLIBS%\Coin3d\include"
   set "COIN3D_DEFINES=COIN_DLL"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\Coin3d\bin"
   
   if exist %EXTERNLIBS%\Coin3d\lib\coin3.lib (
     if "%USE_OPT_LIBS%" == "1" (
        set "COIN3D_LIBS=-L%EXTERNLIBS%\Coin3d\lib -lcoin3"
     ) else (
        set "COIN3D_LIBS=-L%EXTERNLIBS%\Coin3d\lib -lcoin3d"
     )

   ) else (
     if "%USE_OPT_LIBS%" == "1" (
        set "COIN3D_LIBS=-L%EXTERNLIBS%\Coin3d\lib -lcoin2"
     ) else (
        set "COIN3D_LIBS=-L%EXTERNLIBS%\Coin3d\lib -lcoin2d"
     )
   )
)



if not defined SOQT_HOME  (
   set "SOQT_HOME=%COIN3D_HOME%"
   set "SOQT_INCPATH=%COIN3D_HOME%\include"
   set "PATHADD=%PATHADD%;%COIN3D_HOME%\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "SOQT_LIBS=-L%COIN3D_HOME%\lib -lsoqt1"
   ) else (
      set "SOQT_LIBS=-L%COIN3D_HOME%\lib -lsoqt1d"
   )
)


if not defined OIV_HOME (
   set "OIV_HOME=%EXTERNLIBS%\OpenInventor"
   set "OIV_INCPATH=%EXTERNLIBS%\OpenInventor\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\OpenInventor\lib"
   if "%USE_OPT_LIBS%" == "1" (
      set "OIV_LIBS=-L%EXTERNLIBS%\OpenInventor\lib -linventor"
   ) else (
      set "OIV_LIBS=-L%EXTERNLIBS%\OpenInventor\lib -linventorD"
   )
)


if not defined PERFORMER_HOME (
   set "PERFORMER_HOME=%EXTERNLIBS%\Performer"
   set "PFHOME=%EXTERNLIBS%\Performer"
   set "PERFORMER_INCPATH=%EXTERNLIBS%\Performer\Include %EXTERNLIBS%\Performer\Include\Performer"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\Performer\Lib;%EXTERNLIBS%\Performer\Lib\libpfdb"
   set "PERFORMER_LIBS=-L%EXTERNLIBS%\Performer\Lib -llibpf -llibpfdu-util"
)


if not defined PTHREAD_HOME (
   set "PTHREAD_HOME=%EXTERNLIBS%\pthreads"
   set "PTHREAD_INCPATH=%EXTERNLIBS%\pthreads\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\pthreads\lib"
   if "%USE_OPT_LIBS%" == "1" (
      set "PTHREAD_LIBS=-L%EXTERNLIBS%\pthreads\lib -lpthreadVC2"
   ) else (
      set "PTHREAD_LIBS=-L%EXTERNLIBS%\pthreads\lib -lpthreadVC2d"
   )
)


if not defined GLUT_HOME (
   set "GLUT_HOME=%EXTERNLIBS%\glut"
   set "GLUT_INCPATH=%EXTERNLIBS%\glut\include"
   REM set GLUT_DEFINES=GLUT_NO_LIB_PRAGMA
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\glut\lib"
   if "%USE_OPT_LIBS%" == "1" (
      set "GLUT_LIBS=-L%EXTERNLIBS%\glut\lib -lglut"
   ) else (
      set "GLUT_LIBS=-L%EXTERNLIBS%\glut\lib -lglutD"
   )
)



if not defined CURL_HOME (
   set "CURL_HOME=%EXTERNLIBS%\curl"
   set "CURL_INCPATH=%EXTERNLIBS%\curl\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\curl\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "CURL_LIBS=-L%EXTERNLIBS%\curl\lib -llibcurl"
   ) else (
      set "CURL_LIBS=-L%EXTERNLIBS%\curl\lib -llibcurlD"
   )
)


if not defined EXPAT_HOME (
   set "EXPAT_HOME=%EXTERNLIBS%\expat"
   set "EXPAT_INCPATH=%EXTERNLIBS%\expat\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\expat\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "EXPAT_LIBS=-L%EXTERNLIBS%\expat\lib -llibexpat"
   ) else (
      set "EXPAT_LIBS=-L%EXTERNLIBS%\expat\lib -llibexpatD"
   )
)


if not defined GLEW_HOME (
   set "GLEW_HOME=%EXTERNLIBS%\glew"
   set "GLEW_INCPATH=%EXTERNLIBS%\glew\include"
   if "%USE_OPT_LIBS%" == "1" (
       set "GLEW_LIBS=-L%EXTERNLIBS%\glew -lGLEW32"
   ) else (
       set "GLEW_LIBS=-L%EXTERNLIBS%\glew -lGLEW32D"
   )
)


if not defined OPENCV_HOME (
   set "OPENCV_HOME=%EXTERNLIBS%\OpenCV"
   set "OPENCV_INCPATH=%EXTERNLIBS%\OpenCV\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\OpenCV\lib;%EXTERNLIBS%\OpenCV\bin"
   if "%USE_OPT_LIBS%" == "1" (
      REM set "OPENCV_LIBS=-L%EXTERNLIBS%\OpenCV\lib -lcxcore -lcv  -lhighgui -lcvaux -lcvcam"
      set "OPENCV_LIBS=-L%EXTERNLIBS%\OpenCV\lib -lopencv_core220 -lopencv_legacy220  -lopencv_highgui220 -lopencv_imgproc220 -lopencv_video220 -lopencv_features2d220 -lopencv_contrib220 -lopencv_calib3d220 -lopencv_flann220"
   ) else (
      REM set "OPENCV_LIBS=-L%EXTERNLIBS%\OpenCV\lib -lcxcoreD -lcvD  -lhighguiD -lcvauxD -lcvcamD"
      set "OPENCV_LIBS=-L%EXTERNLIBS%\OpenCV\lib -lopencv_core220d -lopencv_legacy220d  -lopencv_highgui220d -lopencv_imgproc220d -lopencv_video220d -lopencv_features2d220d -lopencv_contrib220d -lopencv_calib3d220d -lopencv_flann220d"
   )
)

REM You will probably have to define this in mycommon.bat

if not defined F77_HOME (

     if exist C:\Programme\Intel\Compiler\Fortran\9.1\IA32 (
     
        set F77_HOME=C:\Programme\Intel\Compiler\Fortran\9.1
        set F77_INCPATH=C:\Programme\Intel\Compiler\Fortran\9.1\IA32\Include
        set "PATHADD=%PATHADD%;C:\Programme\Intel\Compiler\Fortran\9.1\IA32\Bin;C:\Programme\Intel\Compiler\C++\9.1\IA32\Bin"
        if "%USE_OPT_LIBS%" == "1" (
           set F77_LIBS=-LC:\Programme\Intel\Compiler\Fortran\9.1\IA32\Lib -llibifcoremd -lifconsol
        ) else (
           set F77_LIBS=-LC:\Programme\Intel\Compiler\Fortran\9.1\IA32\Lib -llibifcoremdd -lifconsol
         )
     ) else if exist C:\Intel\Compiler\11.0\072\fortran (
     
        set F77_HOME=C:\Intel\Compiler\11.0\072\fortran
        set F77_INCPATH=C:\Intel\Compiler\11.0\072\fortran\include
        set "PATHADD=%PATHADD%;C:\Intel\Compiler\11.0\072\fortran\Bin\IA32;C:\Intel\Compiler\11.0\072\cpp\Bin\IA32"
        if "%USE_OPT_LIBS%" == "1" (
           set F77_LIBS=-LC:\Intel\Compiler\11.0\072\fortran\lib\ia32 -llibifcoremd -lifconsol
        ) else (
           set F77_LIBS=-LC:\Intel\Compiler\11.0\072\fortran\lib\ia32 -llibifcoremdd -lifconsol
        )
     ) else if exist "c:\Progra~2\Intel\Compiler\11.0\072\fortran\lib\intel64" (
     
        set F77_HOME="c:\Progra~2\Intel\Compiler\11.0\072\fortran"
        set F77_INCPATH="c:\Progra~2\Intel\Compiler\11.0\072\fortran\include"
        set "PATHADD=%PATHADD%;"c:\Progra~2\Intel\Compiler\11.0\072\fortran\bin\intel64""
        if "%USE_OPT_LIBS%" == "1" (
           set F77_LIBS=-L"c:\Progra~2\Intel\Compiler\11.0\072\fortran\lib\intel64" -llibifcoremd -lifconsol
        ) else (
           set F77_LIBS=-L"c:\Progra~2\Intel\Compiler\11.0\072\fortran\lib\intel64" -llibifcoremdd -lifconsol
        )
     ) else (
REM c:\src\covise/angusopt/lib;c:\src\externlibs\angus/OpenSSL/lib;c:\src\externlibs\angus\Xerces\lib;C:\Programme\Intel\Compiler\Fortran\9.0\IA32\Lib;""
   set F77_HOME=C:\Programme\Intel\Compiler\Fortran\9.0
   set F77_INCPATH=C:\Programme\Intel\Compiler\Fortran\9.0\IA32\Include
   set "PATHADD=%PATHADD%;C:\Programme\Intel\Compiler\Fortran\9.0\IA32\Bin;C:\Programme\Intel\Compiler\C++\9.0\IA32\Bin"
   if "%USE_OPT_LIBS%" == "1" (
      set F77_LIBS=-LC:\Programme\Intel\Compiler\Fortran\9.0\IA32\Lib -llibifcoremd -lifconsol
   ) else (
      set F77_LIBS=-LC:\Programme\Intel\Compiler\Fortran\9.0\IA32\Lib -llibifcoremdd -lifconsol
   )
   )
)

set "HAVE_PTGREY=true"
if /I "%ARCHSUFFIX%" == "angusopt" (
  set "HAVE_PTGREY=false"
) else if /I "%ARCHSUFFIX%" == "angus" (
  set "HAVE_PTGREY=false"
)
if /I "%ARCHSUFFIX%" == "zackelopt" (
  set "HAVE_PTGREY=false"
) else if /I "%ARCHSUFFIX%" == "zackel" (
  set "HAVE_PTGREY=false"
)

if not defined ARTOOLKIT_HOME  (
  if "%HAVE_PTGREY%" == "true" (
    set "PTGREY_HOME=%EXTERNLIBS%\PtGrey"
    set "ARTOOLKIT_HOME=%EXTERNLIBS%\ARToolKit"
    set "ARTOOLKIT_INCPATH=%EXTERNLIBS%\ARToolKit\include %EXTERNLIBS%\PtGrey\include %EXTERNLIBS%\videoInput\include"
    set "ARTOOLKIT_DEFINES=HAVE_AR HAVE_PTGREY"
    set "PATHADD=%PATHADD%;%EXTERNLIBS%\PtGrey\bin"
    
	if EXIST %EXTERNLIBS%\ARToolKit\lib\ARToolKit.lib (
	  if "%USE_OPT_LIBS%" == "1" (
        set "ARTOOLKIT_LIBS=-L%EXTERNLIBS%\ARToolKit\lib -L%EXTERNLIBS%\videoInput\lib -L%EXTERNLIBS%\PtGrey\lib -lARToolKit -lvideoInput -lPGRFlyCapture"
      ) else (
        set "ARTOOLKIT_LIBS=-L%EXTERNLIBS%\ARToolKit\lib -L%EXTERNLIBS%\videoInput\lib -L%EXTERNLIBS%\PtGrey\lib -lARToolKitD -lvideoInputd -lPGRFlyCapture"
      )
	) else (
      if "%USE_OPT_LIBS%" == "1" (
        set "ARTOOLKIT_LIBS=-L%EXTERNLIBS%\ARToolKit\lib -L%EXTERNLIBS%\videoInput\lib -L%EXTERNLIBS%\PtGrey\lib -lAR -lARMulti -lvideoInput -lstrmbase -lstrmiids -lquartz -lPGRFlyCapture"
      ) else (
        set "ARTOOLKIT_LIBS=-L%EXTERNLIBS%\ARToolKit\lib -L%EXTERNLIBS%\videoInput\lib -L%EXTERNLIBS%\PtGrey\lib -lARd -lARMultid -lvideoInputd -lstrmbased -lstrmiids -lquartz -lPGRFlyCapture"
      )
	)
  ) else (
    set "ARTOOLKIT_HOME=%EXTERNLIBS%\ARToolKit"
    set "ARTOOLKIT_INCPATH=%EXTERNLIBS%\ARToolKit\include %EXTERNLIBS%\videoInput\include"
    set "ARTOOLKIT_DEFINES=HAVE_AR"
    set "PATHADD=%PATHADD%;%EXTERNLIBS%\ARToolKit\lib;%EXTERNLIBS%\videoInput\lib"

    if EXIST %EXTERNLIBS%\ARToolKit\lib\ARToolKit.lib (
	  if "%USE_OPT_LIBS%" == "1" (
        set "ARTOOLKIT_LIBS=-L%EXTERNLIBS%\ARToolKit\lib -L%EXTERNLIBS%\videoInput\lib -lARToolKit -lvideoInput"
      ) else (
        set "ARTOOLKIT_LIBS=-L%EXTERNLIBS%\ARToolKit\lib -L%EXTERNLIBS%\videoInput\lib -lARToolKitD -lvideoInputd"
      )
	) else (
      if "%USE_OPT_LIBS%" == "1" (
        set "ARTOOLKIT_LIBS=-L%EXTERNLIBS%\ARToolKit\lib -L%EXTERNLIBS%\videoInput\lib -lAR -lARMulti -lvideoInput -lstrmbase -lstrmiids -lquartz"
      ) else (
        set "ARTOOLKIT_LIBS=-L%EXTERNLIBS%\ARToolKit\lib -L%EXTERNLIBS%\videoInput\lib -lARd -lARMultid -lvideoInputd -lstrmbased -lstrmiids -lquartz"
      )
	)
  )
)

if not defined ARTOOLKITPLUS_HOME  (
  if "%HAVE_PTGREY%" == "true" (
    set "PTGREY_HOME=%EXTERNLIBS%\PtGrey"
    set "ARTOOLKITPLUS_HOME=%EXTERNLIBS%\ARToolKitPlus"
    set "ARTOOLKITPLUS_INCPATH=%EXTERNLIBS%\ARToolKitPlus\include %EXTERNLIBS%\PtGrey\include %EXTERNLIBS%\videoInput\include"
    set "ARTOOLKITPLUS_DEFINES=HAVE_AR HAVE_PTGREY"
    set "PATHADD=%PATHADD%;%EXTERNLIBS%\PtGrey\bin"
	
	REM on some systems, aparently you need -lstrmbase -lstrmiids -lquartz , if you do , tellme, uwe
    
    if "%USE_OPT_LIBS%" == "1" (
      set "ARTOOLKITPLUS_LIBS=-L%EXTERNLIBS%\ARToolKitPlus\lib -L%EXTERNLIBS%\videoInput\lib -L%EXTERNLIBS%\PtGrey\lib -lARToolKitPlusDll -lvideoInput -lPGRFlyCapture"
    ) else (
      set "ARTOOLKITPLUS_LIBS=-L%EXTERNLIBS%\ARToolKitPlus\lib -L%EXTERNLIBS%\videoInput\lib -L%EXTERNLIBS%\PtGrey\lib -lARToolKitPlusDlld -lvideoInputd -lPGRFlyCapture"
    )
  ) else (
    set "ARTOOLKITPLUS_HOME=%EXTERNLIBS%\ARToolKitPlus"
    set "ARTOOLKITPLUS_INCPATH=%EXTERNLIBS%\ARToolKitPlus\include %EXTERNLIBS%\videoInput\include"
    set "ARTOOLKITPLUS_DEFINES=HAVE_AR _USE_DOUBLE ARTOOLKITPLUS_DLL"
    set "PATHADD=%PATHADD%;%EXTERNLIBS%\ARToolKitPlus\lib;%EXTERNLIBS%\videoInput\lib"

    if "%USE_OPT_LIBS%" == "1" (
      set "ARTOOLKITPLUS_LIBS=-L%EXTERNLIBS%\ARToolKitPlus\lib -L%EXTERNLIBS%\videoInput\lib -lARToolKitPlus -lvideoInput"
    ) else (
      set "ARTOOLKITPLUS_LIBS=-L%EXTERNLIBS%\ARToolKitPlus\lib -L%EXTERNLIBS%\videoInput\lib -lARToolKitPlusd -lvideoInputd"
    )
  )
)

if not defined CAL3D_HOME (
   set "CAL3D_HOME=%EXTERNLIBS%\Cal3D"
   set "CAL3D_INCPATH=%EXTERNLIBS%\Cal3D\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\Cal3D\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "CAL3D_LIBS=-L%EXTERNLIBS%\Cal3D\lib -lcal3d"
   ) else (
      set "CAL3D_LIBS=-L%EXTERNLIBS%\Cal3D\lib -lcal3d_D"
   )
)

if not defined OSSIMPLANET_HOME (
   set "OSSIMPLANET_HOME=%EXTERNLIBS%\ossim"
   set "OSSIMPLANET_INCPATH=%EXTERNLIBS%\ossim\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\ossim\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "OSSIMPLANET_LIBS=-L%EXTERNLIBS%\ossim\lib -lossim -lossimPlanet"
   ) else (
      set "OSSIMPLANET_LIBS=-L%EXTERNLIBS%\ossim\lib -lossimD -lossimPlanetD"
   )
)


if not defined OSGEARTH_HOME (
   set "OSGEARTH_HOME=%EXTERNLIBS%\osgEarth"
   set "OSGEARTH_INCPATH=%EXTERNLIBS%\osgEarth\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\geos\bin;%EXTERNLIBS%\osgEarth\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "OSGEARTH_LIBS=-L%EXTERNLIBS%\osgEarth\lib --losgEarth"
   ) else (
      set "OSGEARTH_LIBS=-L%EXTERNLIBS%\osgEarth\lib -losgEarthD"
   )
)


if not defined VISSDK_HOME  (
   set "VISSDK_HOME=%EXTERNLIBS%\VisSDK"
   set "VISSDK_LIBS=-L%EXTERNLIBS%\VisSDK\lib -lVisCoreDB"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\VisSDK\lib"
)

if not defined VTK_VERSION (
   set "VTK_VERSION=5.2"
)

if not defined VTK_HOME (
   set "VTK_HOME=%EXTERNLIBS%\vtk"
   set "VTK_INCPATH=%EXTERNLIBS%\vtk\include\vtk-%VTK_VERSION%"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\vtk\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "VTK_LIBS=-L%EXTERNLIBS%\vtk\lib\vtk-%VTK_VERSION% -lvtkCommon -lvtkFiltering -lvtkIO -lvtkGraphics -lvtkRendering -lvtksys"
   ) else (
      set "VTK_LIBS=-L%EXTERNLIBS%\vtk\lib\vtk-%VTK_VERSION% -lvtkCommonD -lvtkFilteringD -lvtkIOd -lvtkGraphicsD -lvtkRenderingD -lvtksysD"
   )
)

if not defined ITK_HOME  (
   set "ITK_HOME=%EXTERNLIBS%\itk"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\itk\bin"
   set "ITK_INCPATH=%EXTERNLIBS%\itk\include\InsightToolkit %EXTERNLIBS%\itk\include\InsightToolkit\SpatialObject %EXTERNLIBS%\itk\include\InsightToolkit\Common %EXTERNLIBS%\itk\include\InsightToolkit\Algorithms %EXTERNLIBS%\itk\include\InsightToolkit\Utilities %EXTERNLIBS%\itk\include\InsightToolkit\Numerics %EXTERNLIBS%\itk\include\InsightToolkit\Numerics/Statistics %EXTERNLIBS%\itk\include\InsightToolkit\IO %ITK_INCPATH%;%EXTERNLIBS%\itk\include\InsightToolkit\SpatialObject %EXTERNLIBS%\itk\include\InsightToolkit\BasicFilters %EXTERNLIBS%\itk\include\InsightToolkit\Utilities/vxl/vcl  %EXTERNLIBS%\itk\include\InsightToolkit\Utilities/vxl/core"
   if "%USE_OPT_LIBS%" == "1" (
     set "ITK_LIBS=-L%EXTERNLIBS%\itk\lib\InsightToolkit -lsnmpapi -lrpcrt4 -lITKEXPAT -lITKDICOMParser -litkvcl -litkvnl_algo -litkvnl_inst -lITKIO -lITKNumerics -lITKAlgorithms -lITKBasicFilters -lITKStatistics -lITKSpatialObject -lITKFEM -litkzlib -litkv3p_netlib -litkvnl -litkgdcm -litkjpeg8 -litkjpeg12 -litkjpeg16 -litkopenjpeg -lITKCommon -litksys -lITKniftiio -lITKznz -lITKNrrdIO -lITKMetaIO -litktiff -litkpng"
   ) else (
     set "ITK_LIBS=-L%EXTERNLIBS%\itk\lib\InsightToolkit -lsnmpapi -lrpcrt4 -lITKEXPATd -lITKDICOMParserd -litkvcld -litkvnl_algod -litkvnl_instd -lITKIOd -lITKNumericsd -lITKAlgorithmsd -lITKBasicFiltersd -lITKStatisticsd -lITKSpatialObjectd -lITKFEMd -litkzlibd -litkv3p_netlibd -litkvnld -litkgdcmd -litkjpeg8d -litkjpeg12d -litkjpeg16d -litkopenjpegd -lITKCommond -litksysd -lITKniftiiod -lITKznzd -lITKNrrdIOd -lITKMetaIOd -litktiffd -litkpngd"
   )
)


if not defined TCL_HOME  (
   set "TCL_HOME=%EXTERNLIBS%\TCL"
   set "TCL_INCPATH=%EXTERNLIBS%\TCL\include"
   if "%USE_OPT_LIBS%" == "1" (
      set "TCL_LIBS=-L%EXTERNLIBS%\TCL\lib -ltcl85"
   ) else (
      set "TCL_LIBS=-L%EXTERNLIBS%\TCL\lib -ltcl85g"
   )
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\TCL\bin"
)

 
if not defined TK_HOME  ( 
   set "TK_HOME=%EXTERNLIBS%\TCL"
   set "TK_INCPATH=%EXTERNLIBS%\TCL\include"
   if "%USE_OPT_LIBS%" == "1" (
      set "TK_LIBS=-L%EXTERNLIBS%\TCL\lib -ltk85"
   ) else (
      set "TK_LIBS=-L%EXTERNLIBS%\TCL\lib -ltk85g"
   )
)


if not defined VIRVO_HOME  (
   set "VIRVO_HOME=%EXTERNLIBS%\virvo"
   set "VIRVO_INCPATH=%EXTERNLIBS%\virvo\include"
   if "%USE_OPT_LIBS%" == "1" (
      set "VIRVO_LIBS=-L%EXTERNLIBS%\virvo\lib -llibvirvo"
   ) else (
      set "VIRVO_LIBS=-L%EXTERNLIBS%\virvo\lib -llibvirvoD"
   ) 
)

if not defined ICU_HOME  (
   set "ICU_HOME=%EXTERNLIBS%\icu"
   set "ICU_INCPATH=%EXTERNLIBS%\icu\include"
   if "%USE_OPT_LIBS%" == "1" (
      set "ICU_LIBS=-L%EXTERNLIBS%\icu\lib64"
   ) else (
      set "ICU_LIBS=-L%EXTERNLIBS%\icu\lib64"
   ) 
   set "PATH=%PATH%;%EXTERNLIBS%\icu\bin64"
)

if not defined PYTHONHOME  (
   set "PYTHONHOME=%EXTERNLIBS%\..\shared\Python;%EXTERNLIBS%\Python"
   rem PYTHON_HOME is for compiling Python 
   rem  while PYTHONHOME is for executing Python and can consist of
   rem several different paths
   rem set "PYTHONPATH=%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\Python"
   set "PYTHONPATH=%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\Python;%COVISEDIR%\PYTHON\bin;%COVISEDIR%\PYTHON\bin\vr-prepare;%COVISEDIR%\PYTHON\bin\vr-prepare\converters;%COVISEDIR%\PYTHON\bin\vr-prepare\negotiator;%COVISEDIR%\PYTHON\bin\vr-prepare\negotiator\import;%EXTERNLIBS%\pyqt\modules;%EXTERNLIBS%\sip\modules"
   set "PYTHON_INCLUDE=%EXTERNLIBS%\Python\include"
   set "PYTHON_INCPATH=%EXTERNLIBS%\..\shared\Python\include\Python"
   IF /i "%ARCHSUFFIX%" == "vistaopt" (
    set "PYTHON_LIB=%EXTERNLIBS%\Python\PCbuild\python26.lib"
   ) ELSE ( 
    set "PYTHON_LIB=%EXTERNLIBS%\Python\PCbuild\python26_d.lib"
   )
   set "PATH=%PATH%;%EXTERNLIBS%\Python\DLLs;%EXTERNLIBS%\Python;%EXTERNLIBS%\Python\bin"
)

if not defined PYTHON_HOME  (
   set "PYTHON_HOME=%EXTERNLIBS%\python"
   set "PYTHON_INCPATH=%EXTERNLIBS%\python\include"
   if "%USE_OPT_LIBS%" == "1" (
    set "PYTHON_LIBS=%EXTERNLIBS%\Python\libs\python26.lib"
   ) ELSE ( 
    set "PYTHON_LIBS=%EXTERNLIBS%\Python\libs\python26_d.lib"
   )
)

if not defined GDAL_HOME  (
   set "GDAL_HOME=%EXTERNLIBS%\gdal"
   set "GDAL_INCPATH=%EXTERNLIBS%\gdal\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\gdal\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "GDAL_LIBS=-L%EXTERNLIBS%\gdal\lib -lgdal_i"
   ) else (
      set "GDAL_LIBS=-L%EXTERNLIBS%\gdal\lib -lgdal_id"
   ) 
)

if not defined XERCESC_HOME (
    set "XERCESC_HOME=%EXTERNLIBS%\Xerces"
    set "XERCESC_INCPATH=%EXTERNLIBS%\Xerces\include"
    set "PATHADD=%PATHADD%;%EXTERNLIBS%\Xerces\lib"
    if exist %EXTERNLIBS%\Xerces\lib\xerces-c_3.lib (
      REM use xerces version 3
      if "%USE_OPT_LIBS%" == "1" (
        set "XERCESC_LIBS=-L%EXTERNLIBS%\Xerces\lib -lxerces-c_3"
      ) else (
        set "XERCESC_LIBS=-L%EXTERNLIBS%\Xerces\lib -lxerces-c_3D"
      ) 
    ) else (
      REM use xerces version 2
      if "%USE_OPT_LIBS%" == "1" (
         set "XERCESC_LIBS=-L%EXTERNLIBS%\Xerces\lib -lxerces-c_2 -lxerces-depdom_2"
      ) else (
         set "XERCESC_LIBS=-L%EXTERNLIBS%\Xerces\lib -lxerces-c_2D -lxerces-depdom_2D"
      )
    )	
)


if not defined CFX5_UNITS_DIR (
   set "CFX5_UNITS_DIR=%COVISEDIR%\icons"
)

if not defined FARO_HOME  (
   set "FARO_HOME=%EXTERNLIBS%\Faro"
   set "FARO_INCPATH=%EXTERNLIBS%\Faro\Inc"
   set FARO_DEFINES=HAVE_FARO
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\Faro\Bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "FARO_LIBS=-L%EXTERNLIBS%\Faro\lib -lFaroLaserScannerAPI"
   ) else (
      set "FARO_LIBS=-L%EXTERNLIBS%\Faro\lib -lFaroLaserScannerAPI"
   )
)

if not defined WMFSDK_HOME (
   set "WMFSDK_HOME=%EXTERNLIBS%\WMFSDK11"
   set "WMFSDK_INCPATH=%EXTERNLIBS%\WMFSDK11\Include"
   set "WMFSDK_LIBS=-L%EXTERNLIBS%\WMFSDK11\lib -lwmvcore"
   set "WMFSDK_DEFINES=HAVE_WMFSDK"
)

if not defined PCAN_HOME ( 
   if exist %EXTERNLIBS%\PCAN-Light\nul (
     set "PCAN_HOME=%EXTERNLIBS%\PCAN-Light"
     set "PATHADD=%PATHADD%;%EXTERNLIBS%\PCAN-Light"
     set "PCAN_INCPATH=%EXTERNLIBS%\PCAN-Light\Api"
     set "PCAN_LIBS=-L%EXTERNLIBS%\PCAN-Light\Lib\VC -lPcan_pci"
     set PCAN_DEFINES=HAVE_PCAN
   )
)

if not defined WIIUSE_HOME (
   set "WIIUSE_HOME=%EXTERNLIBS%\wiiuse"
   set "WIIUSE_LIBS=-L%EXTERNLIBS%\wiiuse -lwiiuse"
   set "WIIUSE_INCPATH=%EXTERNLIBS%\wiiuse"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\wiiuse"
)
     
if not defined WIIUSE_HOME (
   set "WIIUSE_HOME=%EXTERNLIBS%\wiiuse"
   set "WIIUSE_LIBS=-L%EXTERNLIBS%\wiiuse\lib -lwiiuse"
   set "WIIUSE_INCPATH=%EXTERNLIBS%\wiiuse\include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\wiiuse"
)
    
if not defined WIIYOURSELF_HOME (
   set "WIIYOURSELF_HOME=%EXTERNLIBS%\wiimote"
   set "WIIYOURSELF_INCPATH=%EXTERNLIBS%\wiimote\include"
   if "%USE_OPT_LIBS%" == "1" (
       set "WIIYOURSELF_LIBS=-L%EXTERNLIBS%\wiimote\lib -lwiimote -lWinmm -lSetupapi"
   ) else (
       set "WIIYOURSELF_LIBS=-L%EXTERNLIBS%\wiimote\lib -lwiimoteD  -lWinmm -lSetupapi"
   ) 
)

if not defined WIIMOTELIB_HOME (
   set "WIIMOTELIB_HOME=%EXTERNLIBS%\WiimoteLib"
   set "WIIMOTELIB_INCPATH=%EXTERNLIBS%\WiimoteLib\include"
   if "%USE_OPT_LIBS%" == "1" (
      set "WIIMOTELIB_LIBS=-L%EXTERNLIBS%\WiimoteLib\lib -lWiimoteLib"
   ) else (
      set "WIIMOTELIB_LIBS=-L%EXTERNLIBS%\WiimoteLib\lib -lWiimoteLibd"
   )
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\WiimoteLib\bin"
)

rem all path additions have been added to PATHADD
rem as expanding a PATH containing "(x86)" terminates the parentheses opened after if
set "PATH=%PATHPRE%;%PATH%;%QT_HOME%\bin;%PATHADD%"
