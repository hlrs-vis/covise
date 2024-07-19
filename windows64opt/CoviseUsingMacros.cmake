file(GLOB USING_FILES "${COVISEDIR}/cmake/Using/Use*.cmake" "${COVISEDIR}/share/covise/cmake/Using/Use*.cmake")
foreach(F ${USING_FILES})
  include(${F})
endforeach(F ${USING_FILES})

MACRO(USING)

  SET(optional FALSE)
  STRING (REGEX MATCHALL "(^|[^a-zA-Z0-9_])optional($|[^a-zA-Z0-9_])" optional "${ARGV}")

  FOREACH(feature ${ARGV})

    STRING (REGEX MATCH "^[a-zA-Z0-9_]+" use "${feature}")

    STRING (REGEX MATCH ":[a-zA-Z0-9_]+$" component "${feature}")

    STRING (REGEX MATCH "[a-zA-Z0-9_]+$" component "${component}")

    STRING (TOUPPER ${use} use)
    IF(use STREQUAL ABAQUS)
      USE_ABAQUS(${component} ${optional})
    ENDIF(use STREQUAL ABAQUS)

    IF(use STREQUAL ALVAR)
      USE_ALVAR(${component} ${optional})
    ENDIF(use STREQUAL ALVAR)

    IF(use STREQUAL ARTOOLKITPLUS)
      USE_ARTOOLKITPLUS(${component} ${optional})
    ENDIF(use STREQUAL ARTOOLKITPLUS)

    IF(use STREQUAL ARTOOLKIT)
      USE_ARTOOLKIT(${component} ${optional})
    ENDIF(use STREQUAL ARTOOLKIT)

    IF(use STREQUAL ASSIMP)
      USE_ASSIMP(${component} ${optional})
    ENDIF(use STREQUAL ASSIMP)

    IF(use STREQUAL BIFBOF)
      USE_BIFBOF(${component} ${optional})
    ENDIF(use STREQUAL BIFBOF)

    IF(use STREQUAL BOOST)
      USE_BOOST(${component} ${optional})
    ENDIF(use STREQUAL BOOST)

    IF(use STREQUAL BULLET)
      USE_BULLET(${component} ${optional})
    ENDIF(use STREQUAL BULLET)

    IF(use STREQUAL CEF)
      USE_CEF(${component} ${optional})
    ENDIF(use STREQUAL CEF)

    IF(use STREQUAL CGNS)
      USE_CGNS(${component} ${optional})
    ENDIF(use STREQUAL CGNS)

    IF(use STREQUAL CUDPP)
      USE_CUDPP(${component} ${optional})
    ENDIF(use STREQUAL CUDPP)

    IF(use STREQUAL CURL)
      USE_CURL(${component} ${optional})
    ENDIF(use STREQUAL CURL)

    IF(use STREQUAL CAL3D)
      USE_CAL3D(${component} ${optional})
    ENDIF(use STREQUAL CAL3D)

    IF(use STREQUAL CAVEUI)
      USE_CAVEUI(${component} ${optional})
    ENDIF(use STREQUAL CAVEUI)

    IF(use STREQUAL CG)
      USE_CG(${component} ${optional})
    ENDIF(use STREQUAL CG)

    IF(use STREQUAL CURSES)
      USE_CURSES(${component} ${optional})
    ENDIF(use STREQUAL CURSES)

    IF(use STREQUAL DRACO)
      USE_DRACO(${component} ${optional})
    ENDIF(use STREQUAL DRACO)

    IF(use STREQUAL DART)
      USE_DART(${component} ${optional})
    ENDIF(use STREQUAL DART)

    IF(use STREQUAL E57)
      USE_E57(${component} ${optional})
    ENDIF(use STREQUAL E57)

    IF(use STREQUAL EIGEN)
      USE_EIGEN(${component} ${optional})
    ENDIF(use STREQUAL EIGEN)

    IF(use STREQUAL EMBREE3)
      USE_EMBREE3(${component} ${optional})
    ENDIF(use STREQUAL EMBREE3)

    IF(use STREQUAL FFMPEG)
      USE_FFMPEG(${component} ${optional})
    ENDIF(use STREQUAL FFMPEG)

    IF(use STREQUAL FFTW)
      USE_FFTW(${component} ${optional})
    ENDIF(use STREQUAL FFTW)

    IF(use STREQUAL FLEX)
      USE_FLEX(${component} ${optional})
    ENDIF(use STREQUAL FLEX)

    IF(use STREQUAL FMOD)
      USE_FMOD(${component} ${optional})
    ENDIF(use STREQUAL FMOD)

    IF(use STREQUAL FREEIMAGE)
      USE_FREEIMAGE(${component} ${optional})
    ENDIF(use STREQUAL FREEIMAGE)

    IF(use STREQUAL FREETYPE)
      USE_FREETYPE(${component} ${optional})
    ENDIF(use STREQUAL FREETYPE)

    IF(use STREQUAL FORTRAN)
      USE_FORTRAN(${component} ${optional})
    ENDIF(use STREQUAL FORTRAN)

    IF(use STREQUAL GDAL)
      USE_GDAL(${component} ${optional})
    ENDIF(use STREQUAL GDAL)

    IF(use STREQUAL GLEW)
      USE_GLEW(${component} ${optional})
    ENDIF(use STREQUAL GLEW)

    IF(use STREQUAL GLUT)
      USE_GLUT(${component} ${optional})
    ENDIF(use STREQUAL GLUT)

    IF(use STREQUAL GSOAP)
      USE_GSOAP(${component} ${optional})
    ENDIF(use STREQUAL GSOAP)

    IF(use STREQUAL GEOTIFF)
      USE_GEOTIFF(${component} ${optional})
    ENDIF(use STREQUAL GEOTIFF)

    IF(use STREQUAL HDF5)
      USE_HDF5(${component} ${optional})
    ENDIF(use STREQUAL HDF5)

    IF(use STREQUAL IFCPP)
      USE_IFCPP(${component} ${optional})
    ENDIF(use STREQUAL IFCPP)

    IF(use STREQUAL IK)
      USE_IK(${component} ${optional})
    ENDIF(use STREQUAL IK)

    IF(use STREQUAL ITK)
      USE_ITK(${component} ${optional})
    ENDIF(use STREQUAL ITK)

    IF(use STREQUAL JPEG)
      USE_JPEG(${component} ${optional})
    ENDIF(use STREQUAL JPEG)

    IF(use STREQUAL JSBSIM)
      USE_JSBSIM(${component} ${optional})
    ENDIF(use STREQUAL JSBSIM)

    IF(use STREQUAL JT)
      USE_JT(${component} ${optional})
    ENDIF(use STREQUAL JT)

    IF(use STREQUAL JPEGTURBO)
      USE_JPEGTURBO(${component} ${optional})
    ENDIF(use STREQUAL JPEGTURBO)

    IF(use STREQUAL LAMURE)
      USE_LAMURE(${component} ${optional})
    ENDIF(use STREQUAL LAMURE)

    IF(use STREQUAL LIBUSB)
      USE_LIBUSB(${component} ${optional})
    ENDIF(use STREQUAL LIBUSB)

    IF(use STREQUAL LIBUSB1)
      USE_LIBUSB1(${component} ${optional})
    ENDIF(use STREQUAL LIBUSB1)

    IF(use STREQUAL LINPHONEDESKTOP)
      USE_LINPHONEDESKTOP(${component} ${optional})
    ENDIF(use STREQUAL LINPHONEDESKTOP)

    IF(use STREQUAL MICROHTTPD)
      USE_MICROHTTPD(${component} ${optional})
    ENDIF(use STREQUAL MICROHTTPD)

    IF(use STREQUAL MIDIFILE)
      USE_MIDIFILE(${component} ${optional})
    ENDIF(use STREQUAL MIDIFILE)

    IF(use STREQUAL MPI)
      USE_MPI(${component} ${optional})
    ENDIF(use STREQUAL MPI)

    IF(use STREQUAL MOTIF)
      USE_MOTIF(${component} ${optional})
    ENDIF(use STREQUAL MOTIF)

    IF(use STREQUAL NATNET)
      USE_NATNET(${component} ${optional})
    ENDIF(use STREQUAL NATNET)

    IF(use STREQUAL NETCDF)
      USE_NETCDF(${component} ${optional})
    ENDIF(use STREQUAL NETCDF)

    IF(use STREQUAL OPEN62541)
      USE_OPEN62541(${component} ${optional})
    ENDIF(use STREQUAL OPEN62541)

    IF(use STREQUAL OPENTHREADS)
      USE_OPENTHREADS(${component} ${optional})
    ENDIF(use STREQUAL OPENTHREADS)

    IF(use STREQUAL OSC)
      USE_OSC(${component} ${optional})
    ENDIF(use STREQUAL OSC)

    IF(use STREQUAL OSGTERRAIN)
      USE_OSGTERRAIN(${component} ${optional})
    ENDIF(use STREQUAL OSGTERRAIN)

    IF(use STREQUAL OSVR)
      USE_OSVR(${component} ${optional})
    ENDIF(use STREQUAL OSVR)

    IF(use STREQUAL OVR)
      USE_OVR(${component} ${optional})
    ENDIF(use STREQUAL OVR)

    IF(use STREQUAL OPENCRG)
      USE_OPENCRG(${component} ${optional})
    ENDIF(use STREQUAL OPENCRG)

    IF(use STREQUAL OPENCV)
      USE_OPENCV(${component} ${optional})
    ENDIF(use STREQUAL OPENCV)

    IF(use STREQUAL OPENCV2)
      USE_OPENCV2(${component} ${optional})
    ENDIF(use STREQUAL OPENCV2)

    IF(use STREQUAL OPENGL)
      USE_OPENGL(${component} ${optional})
    ENDIF(use STREQUAL OPENGL)

    IF(use STREQUAL OPENNURBS)
      USE_OPENNURBS(${component} ${optional})
    ENDIF(use STREQUAL OPENNURBS)

    IF(use STREQUAL OPENPASS)
      USE_OPENPASS(${component} ${optional})
    ENDIF(use STREQUAL OPENPASS)

    IF(use STREQUAL OPENSCENARIO)
      USE_OPENSCENARIO(${component} ${optional})
    ENDIF(use STREQUAL OPENSCENARIO)

    IF(use STREQUAL OPENVR)
      USE_OPENVR(${component} ${optional})
    ENDIF(use STREQUAL OPENVR)

    IF(use STREQUAL OPENXR)
      USE_OPENXR(${component} ${optional})
    ENDIF(use STREQUAL OPENXR)

    IF(use STREQUAL OSGBULLET)
      USE_OSGBULLET(${component} ${optional})
    ENDIF(use STREQUAL OSGBULLET)

    IF(use STREQUAL OSGCAL)
      USE_OSGCAL(${component} ${optional})
    ENDIF(use STREQUAL OSGCAL)

    IF(use STREQUAL OSGEARTH)
      USE_OSGEARTH(${component} ${optional})
    ENDIF(use STREQUAL OSGEARTH)

    IF(use STREQUAL OSGEPHEMERIS)
      USE_OSGEPHEMERIS(${component} ${optional})
    ENDIF(use STREQUAL OSGEPHEMERIS)

    IF(use STREQUAL OSGPHYSX)
      USE_OSGPHYSX(${component} ${optional})
    ENDIF(use STREQUAL OSGPHYSX)

    IF(use STREQUAL OSGQT)
      USE_OSGQT(${component} ${optional})
    ENDIF(use STREQUAL OSGQT)

    IF(use STREQUAL OSSIMPLANET)
      USE_OSSIMPLANET(${component} ${optional})
    ENDIF(use STREQUAL OSSIMPLANET)

    IF(use STREQUAL PCL)
      USE_PCL(${component} ${optional})
    ENDIF(use STREQUAL PCL)

    IF(use STREQUAL PNG)
      USE_PNG(${component} ${optional})
    ENDIF(use STREQUAL PNG)

    IF(use STREQUAL PROJ4)
      USE_PROJ4(${component} ${optional})
    ENDIF(use STREQUAL PROJ4)

    IF(use STREQUAL PHYSX)
      USE_PHYSX(${component} ${optional})
    ENDIF(use STREQUAL PHYSX)

    IF(use STREQUAL PTHREADS)
      USE_PTHREADS(${component} ${optional})
    ENDIF(use STREQUAL PTHREADS)

    IF(use STREQUAL RBDL)
      USE_RBDL(${component} ${optional})
    ENDIF(use STREQUAL RBDL)

    IF(use STREQUAL REVIT)
      USE_REVIT(${component} ${optional})
    ENDIF(use STREQUAL REVIT)

    IF(use STREQUAL ROADTERRAIN)
      USE_ROADTERRAIN(${component} ${optional})
    ENDIF(use STREQUAL ROADTERRAIN)

    IF(use STREQUAL SDL2)
      USE_SDL2(${component} ${optional})
    ENDIF(use STREQUAL SDL2)

    IF(use STREQUAL SISL)
      USE_SISL(${component} ${optional})
    ENDIF(use STREQUAL SISL)

    IF(use STREQUAL SLAM6D)
      USE_SLAM6D(${component} ${optional})
    ENDIF(use STREQUAL SLAM6D)

    IF(use STREQUAL SNAPPY)
      USE_SNAPPY(${component} ${optional})
    ENDIF(use STREQUAL SNAPPY)

    IF(use STREQUAL SIXENSE)
      USE_SIXENSE(${component} ${optional})
    ENDIF(use STREQUAL SIXENSE)

    IF(use STREQUAL SPEEX)
      USE_SPEEX(${component} ${optional})
    ENDIF(use STREQUAL SPEEX)

    IF(use STREQUAL STEEREO)
      USE_STEEREO(${component} ${optional})
    ENDIF(use STREQUAL STEEREO)

    IF(use STREQUAL SURFACE)
      USE_SURFACE(${component} ${optional})
    ENDIF(use STREQUAL SURFACE)

    IF(use STREQUAL TBB)
      USE_TBB(${component} ${optional})
    ENDIF(use STREQUAL TBB)

    IF(use STREQUAL TIFF)
      USE_TIFF(${component} ${optional})
    ENDIF(use STREQUAL TIFF)

    IF(use STREQUAL TCL)
      USE_TCL(${component} ${optional})
    ENDIF(use STREQUAL TCL)

    IF(use STREQUAL TINYGLTF)
      USE_TINYGLTF(${component} ${optional})
    ENDIF(use STREQUAL TINYGLTF)

    IF(use STREQUAL TRAFFICSIMULATION)
      USE_TRAFFICSIMULATION(${component} ${optional})
    ENDIF(use STREQUAL TRAFFICSIMULATION)

    IF(use STREQUAL V8)
      USE_V8(${component} ${optional})
    ENDIF(use STREQUAL V8)

    IF(use STREQUAL VRML)
      USE_VRML(${component} ${optional})
    ENDIF(use STREQUAL VRML)

    IF(use STREQUAL VRPN)
      USE_VRPN(${component} ${optional})
    ENDIF(use STREQUAL VRPN)

    IF(use STREQUAL VTK)
      USE_VTK(${component} ${optional})
    ENDIF(use STREQUAL VTK)

    IF(use STREQUAL VEHICLEUTIL)
      USE_VEHICLEUTIL(${component} ${optional})
    ENDIF(use STREQUAL VEHICLEUTIL)

    IF(use STREQUAL VIDEOINPUT)
      USE_VIDEOINPUT(${component} ${optional})
    ENDIF(use STREQUAL VIDEOINPUT)

    IF(use STREQUAL VIRVO)
      USE_VIRVO(${component} ${optional})
    ENDIF(use STREQUAL VIRVO)

    IF(use STREQUAL VISIONARAY)
      USE_VISIONARAY(${component} ${optional})
    ENDIF(use STREQUAL VISIONARAY)

    IF(use STREQUAL WIIYOURSELF)
      USE_WIIYOURSELF(${component} ${optional})
    ENDIF(use STREQUAL WIIYOURSELF)

    IF(use STREQUAL WIRINGPI)
      USE_WIRINGPI(${component} ${optional})
    ENDIF(use STREQUAL WIRINGPI)

    IF(use STREQUAL XERCESC)
      USE_XERCESC(${component} ${optional})
    ENDIF(use STREQUAL XERCESC)

    IF(use STREQUAL ZLIB)
      USE_ZLIB(${component} ${optional})
    ENDIF(use STREQUAL ZLIB)

    IF(use STREQUAL ZSPACE)
      USE_ZSPACE(${component} ${optional})
    ENDIF(use STREQUAL ZSPACE)

  ENDFOREACH(feature)

ENDMACRO(USING)
