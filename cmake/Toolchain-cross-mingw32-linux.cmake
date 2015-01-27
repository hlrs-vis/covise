# the name of the target operating system
SET(CMAKE_SYSTEM_NAME Windows)

# require at least 2.8.10, as we had problems with some older 2.8 variants (2.8.10.2 tested)
cmake_minimum_required(VERSION 2.8.10)

# Choose an appropriate compiler prefix

# here is the target environment located
SET(USER_ROOT_PATH $ENV{EXTERNLIBS})

IF(APPLE)
   set(COMPILER_PREFIX "i686-w64-mingw32")
   SET(CMAKE_FIND_ROOT_PATH $ENV{EXTERNLIBS}/darwin-${COMPILER_PREFIX} ${USER_ROOT_PATH})
ELSE()
   # for classical mingw32
   # see http://www.mingw.org/
   #set(COMPILER_PREFIX "i586-mingw32msvc")
   IF(EXISTS "/usr/i686-w64-mingw32/")
      # for 32 or 64 bits mingw-w64
      # see http://mingw-w64.sourceforge.net/
      set(COMPILER_PREFIX "i686-w64-mingw32")
      #set(COMPILER_PREFIX "x86_64-w64-mingw32"
      SET(CMAKE_FIND_ROOT_PATH /usr/${COMPILER_PREFIX} ${USER_ROOT_PATH})
  #ELSEIF(EXISTS "/usr/i686-pc-mingw32/")
      # for Red Hat/Fedora
      #set(COMPILER_PREFIX "i686-pc-mingw32")
      #SET(CMAKE_FIND_ROOT_PATH /usr/${COMPILER_PREFIX} ${USER_ROOT_PATH})
   ELSE()
      # try mingw from externlibs
      set(COMPILER_PREFIX "i686-w64-mingw32")
      SET(CMAKE_FIND_ROOT_PATH $ENV{EXTERNLIBS}/${COMPILER_PREFIX} ${USER_ROOT_PATH})
   ENDIF()
ENDIF()

# which compilers to use
find_program(CMAKE_RC_COMPILER NAMES ${COMPILER_PREFIX}-windres)
find_program(CMAKE_C_COMPILER NAMES ${COMPILER_PREFIX}-gcc)
find_program(CMAKE_CXX_COMPILER NAMES ${COMPILER_PREFIX}-g++)
find_program(CMAKE_Fortran_COMPILER NAMES ${COMPILER_PREFIX}-gfortran)

execute_process(COMMAND ${CMAKE_C_COMPILER} -print-sysroot
   OUTPUT_VARIABLE MINGW_SYSROOT_BASE OUTPUT_STRIP_TRAILING_WHITESPACE)
SET(MINGW_SYSROOT "${MINGW_SYSROOT_BASE}/mingw")

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search 
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)

# Qt stuff ...
#set(QT_QTCORE_INCLUDE_DIR ${USER_ROOT_PATH}/qt4/include/QtCore)
set(QT_QTCORE_LIBRARY_RELEASE ${USER_ROOT_PATH}/qt4/lib/libQtCore4.a)
#set(QT_QTGUI_INCLUDE_DIR ${USER_ROOT_PATH}/qt4/include/QtGui)
set(QT_QTGUI_LIBRARY_RELEASE ${USER_ROOT_PATH}/qt4/lib/libQtGui4.a)
SET(QT_LIBRARY_DIR ${USER_ROOT_PATH}/qt4/lib)
SET(QT_INCLUDE_DIR ${USER_ROOT_PATH}/qt4/include)
IF(APPLE)
   SET(QT_MOC_EXECUTABLE ${USER_ROOT_PATH}/qt4/bin/darwin-moc)
   SET(QT_RCC_EXECUTABLE ${USER_ROOT_PATH}/qt4/bin/darwin-rcc)
   SET(QT_QMAKE_EXECUTABLE ${USER_ROOT_PATH}/qt4/bin/darwin-qmake)
   SET(QT_UIC_EXECUTABLE ${USER_ROOT_PATH}/qt4/bin/darwin-uic)
ELSE()
   SET(QT_MOC_EXECUTABLE ${USER_ROOT_PATH}/qt4/bin/linux-moc)
   SET(QT_RCC_EXECUTABLE ${USER_ROOT_PATH}/qt4/bin/linux-rcc)
   SET(QT_QMAKE_EXECUTABLE ${USER_ROOT_PATH}/qt4/bin/linux-qmake)
   SET(QT_UIC_EXECUTABLE ${USER_ROOT_PATH}/qt4/bin/linux-uic)
ENDIF()

# MNG
SET(MNG_LIBRARY ${USER_ROOT_PATH}/mng/lib/libmng.a)

ADD_DEFINITIONS(-DMINGW)
