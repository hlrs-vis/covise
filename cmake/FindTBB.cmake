## ======================================================================== ##
## Copyright 2009-2015 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

find_path(
    TBB_ROOT include/tbb/task_group.h
    DOC "Root of TBB installation"
    PATHS ${PROJECT_SOURCE_DIR}/tbb
          "C:/Program Files (x86)/Intel/Composer XE/tbb"
          /opt/intel/composerxe/tbb
          /opt/homebrew/opt/tbb
          /usr/local/opt/tbb
          $ENV{EXTERNLIBS}/tbb
          /usr)
#UNSET(TBB_INCLUDE_DIR CACHE)
#UNSET(TBB_LIBRARY CACHE)
#UNSET(TBB_LIBRARY_MALLOC CACHE)

include(FindPackageHandleStandardArgs)

if(WIN32)

    #IF (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(TBB_ARCH intel64)
    #ELSE()
    #  SET(TBB_ARCH ia32)
    #ENDIF()

    if(MSVC10)
        set(TBB_VCVER vc10)
    elseif(MSVC11)
        set(TBB_VCVER vc11)
    elseif(MSVC12)
        set(TBB_VCVER vc12)
    elseif(MSVC13)
        set(TBB_VCVER vc13)
    else()
        set(TBB_VCVER vc14)
    endif()

    set(TBB_LIBDIR ${TBB_ROOT}/lib/${TBB_ARCH}/${TBB_VCVER})
    set(TBB_BINDIR ${TBB_ROOT}/bin/${TBB_ARCH}/${TBB_VCVER})

    find_path(
        TBB_INCLUDE_DIR tbb/task_group.h
        PATHS ${TBB_ROOT}/include
        NO_DEFAULT_PATH)
    find_library(
        TBB_LIBRARY tbb
        PATHS ${TBB_LIBDIR}
        NO_DEFAULT_PATH)
    find_library(
        TBB_LIBRARY_MALLOC tbbmalloc
        PATHS ${TBB_LIBDIR}
        NO_DEFAULT_PATH)
    find_library(
        TBB_LIBRARY_DEBUG tbb_debug
        PATHS ${TBB_LIBDIR}
        NO_DEFAULT_PATH)
    find_library(
        TBB_LIBRARY_MALLOC_DEBUG tbbmalloc_debug
        PATHS ${TBB_LIBDIR}
        NO_DEFAULT_PATH)

    if(NOT TBB_LIBRARY)
        set(TBB_LIBDIR ${TBB_ROOT}/lib)
        set(TBB_DEBUGLIBDIR ${TBB_ROOT}/debug/lib)
        set(TBB_BINDIR ${TBB_ROOT}/bin)

        find_library(
            TBB_LIBRARY tbb
            PATHS ${TBB_LIBDIR}
            NO_DEFAULT_PATH)
        find_library(
            TBB_LIBRARY_MALLOC tbbmalloc
            PATHS ${TBB_LIBDIR}
            NO_DEFAULT_PATH)
        find_library(
            TBB_LIBRARY_DEBUG tbb_debug
            PATHS ${TBB_LIBDIR} ${TBB_DEBUGLIBDIR}
            NO_DEFAULT_PATH)
        find_library(
            TBB_LIBRARY_MALLOC_DEBUG tbbmalloc_debug
            PATHS ${TBB_LIBDIR} ${TBB_DEBUGLIBDIR}
            NO_DEFAULT_PATH)
    endif()

    find_package_handle_standard_args(
        TBB
        DEFAULT_MSG
        TBB_INCLUDE_DIR
        TBB_LIBRARY
        TBB_LIBRARY_MALLOC
        TBB_LIBRARY_DEBUG
        TBB_LIBRARY_MALLOC_DEBUG)

    if(TBB_FOUND)
        set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
        set(TBB_LIBRARIES optimized ${TBB_LIBRARY} ${TBB_LIBRARY_MALLOC} debug ${TBB_LIBRARY_DEBUG} ${TBB_LIBRARY_MALLOC_DEBUG})
    endif()

    if(TBB_INCLUDE_DIR
       AND TBB_LIBRARY_MIC
       AND TBB_LIBRARY_MALLOC_MIC)
        set(TBB_FOUND_MIC TRUE)
        set(TBB_INCLUDE_DIRS_MIC ${TBB_INCLUDE_DIR_MIC})
        set(TBB_LIBRARIES_MIC ${TBB_LIBRARY_MIC} ${TBB_LIBRARY_MALLOC_MIC})
    endif()

    mark_as_advanced(TBB_INCLUDE_DIR)
    mark_as_advanced(TBB_LIBRARY)
    mark_as_advanced(TBB_LIBRARY_MALLOC)
    mark_as_advanced(TBB_LIBRARY_DEBUG)
    mark_as_advanced(TBB_LIBRARY_MALLOC_DEBUG)

else()

    if(APPLE)
        if(ENABLE_INSTALLER)
            find_path(TBB_INCLUDE_DIR tbb/task_group.h)
            find_library(TBB_LIBRARY tbb)
            find_library(TBB_LIBRARY_MALLOC tbbmalloc)
        else()
            find_path(
                TBB_INCLUDE_DIR tbb/task_group.h
                PATHS ${TBB_ROOT}/include
                NO_DEFAULT_PATH)
            find_library(
                TBB_LIBRARY tbb
                PATHS ${TBB_ROOT}/lib
                NO_DEFAULT_PATH)
            find_library(
                TBB_LIBRARY_MALLOC tbbmalloc
                PATHS ${TBB_ROOT}/lib
                NO_DEFAULT_PATH)
        endif()
    else()

        find_path(
            TBB_INCLUDE_DIR tbb/task_group.h
            PATHS ${TBB_ROOT}/include
            NO_DEFAULT_PATH)
        find_library(
            TBB_LIBRARY tbb
            PATHS ${TBB_ROOT}/lib ${TBB_ROOT}/lib64 ${TBB_ROOT}/lib/x86_64-linux-gnu ${TBB_ROOT}/lib/intel64/gcc4.4
            NO_DEFAULT_PATH)
        find_library(
            TBB_LIBRARY_MALLOC tbbmalloc
            PATHS ${TBB_ROOT}/lib ${TBB_ROOT}/lib64 ${TBB_ROOT}/lib/x86_64-linux-gnu ${TBB_ROOT}/lib/intel64/gcc4.4
            NO_DEFAULT_PATH)
    endif()

    find_path(
        TBB_INCLUDE_DIR_MIC tbb/task_group.h
        PATHS ${TBB_ROOT}/include
        NO_DEFAULT_PATH)
    find_library(
        TBB_LIBRARY_MIC tbb
        PATHS ${TBB_ROOT}/lib/mic
        NO_DEFAULT_PATH)
    find_library(
        TBB_LIBRARY_MALLOC_MIC tbbmalloc
        PATHS ${TBB_ROOT}/lib/mic
        NO_DEFAULT_PATH)

    mark_as_advanced(TBB_INCLUDE_DIR_MIC)
    mark_as_advanced(TBB_LIBRARY_MIC)
    mark_as_advanced(TBB_LIBRARY_MALLOC_MIC)

    find_package_handle_standard_args(TBB DEFAULT_MSG TBB_INCLUDE_DIR TBB_LIBRARY TBB_LIBRARY_MALLOC)

    if(TBB_FOUND)
        set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
        set(TBB_LIBRARIES ${TBB_LIBRARY} ${TBB_LIBRARY_MALLOC})
    endif()

    if(TBB_INCLUDE_DIR
       AND TBB_LIBRARY_MIC
       AND TBB_LIBRARY_MALLOC_MIC)
        set(TBB_FOUND_MIC TRUE)
        set(TBB_INCLUDE_DIRS_MIC ${TBB_INCLUDE_DIR_MIC})
        set(TBB_LIBRARIES_MIC ${TBB_LIBRARY_MIC} ${TBB_LIBRARY_MALLOC_MIC})
    endif()

    mark_as_advanced(TBB_INCLUDE_DIR)
    mark_as_advanced(TBB_LIBRARY)
    mark_as_advanced(TBB_LIBRARY_MALLOC)

endif()

##############################################################
# Install TBB
##############################################################

if(FALSE) # not building -> not installing
    if(WIN32)
        install(
            PROGRAMS ${TBB_BINDIR}/tbb.dll ${TBB_BINDIR}/tbbmalloc.dll
            DESTINATION bin
            COMPONENT tutorials)
        install(
            PROGRAMS ${TBB_BINDIR}/tbb.dll ${TBB_BINDIR}/tbbmalloc.dll
            DESTINATION lib
            COMPONENT libraries)
    elseif(APPLE)
        if(NOT ENABLE_INSTALLER)
            install(
                PROGRAMS ${TBB_ROOT}/lib/libc++/libtbb.dylib ${TBB_ROOT}/lib/libc++/libtbbmalloc.dylib
                DESTINATION lib
                COMPONENT libraries)
        endif()
    else()
        if(NOT ENABLE_INSTALLER)
            install(
                PROGRAMS ${TBB_ROOT}/lib/intel64/gcc4.4/libtbb.so.2 ${TBB_ROOT}/lib/intel64/gcc4.4/libtbbmalloc.so.2
                DESTINATION lib
                COMPONENT libraries)
        endif()
    endif()
endif()
