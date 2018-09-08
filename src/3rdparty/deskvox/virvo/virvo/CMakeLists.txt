if(DESKVOX_USE_ZEROCONF)
  find_package(Bonjour)
endif()
find_package(Boost COMPONENTS chrono date_time filesystem serialization system thread REQUIRED)
find_package(Cg)
find_package(FFMPEG)
find_package(GLEW REQUIRED)
find_package(JPEGTURBO)
find_package(Nifti)
find_package(NORM)
find_package(OpenGL REQUIRED)
find_package(PNG)
find_package(ZLIB)
find_package(Protokit)
find_package(Pthreads)
find_package(SNAPPY)
find_package(Teem)
find_package(VolPack)
find_package(X11)
if(DESKVOX_USE_GDCM)
  find_package(GDCM)
  if (GDCM_FOUND)
    include(${GDCM_USE_FILE})
  endif()
  deskvox_use_package(GDCM)
endif()

deskvox_use_package(Bonjour)
deskvox_use_package(Boost)
deskvox_use_package(Cg INCDIRS ${CG_INCLUDE_DIRS} LIBS ${CG_LIBRARIES})
if(DESKVOX_USE_CUDA)
    find_package(CUDA)
    deskvox_use_package(CUDA INCDIRS ${CUDA_TOOLKIT_INCLUDE})
endif(DESKVOX_USE_CUDA)
deskvox_use_package(FFMPEG)
deskvox_use_package(GLEW)
deskvox_use_package(JPEGTURBO)
deskvox_use_package(Nifti)
deskvox_use_package(NORM)
deskvox_use_package(OpenGL)
deskvox_use_package(PNG)
deskvox_use_package(ZLIB)
deskvox_use_package(Protokit)
deskvox_use_package(Pthreads)
deskvox_use_package(SNAPPY)
deskvox_use_package(Teem)
deskvox_use_package(VolPack)
deskvox_use_package(X11)


if(MSVC)
    # Disable Boost autolinking when compiling w/ msvc
    add_definitions(-DBOOST_ALL_NO_LIB)
endif()


#---------------------------------------------------------------------------------------------------
# Create config file
#


foreach(p ${__DESKVOX_USED_PACKAGES})
  set(VV_HAVE_${p} 1)
endforeach()

configure_file(${VIRVO_INCLUDE_DIR}/virvo/vvconfig.h.in ${VIRVO_CONFIG_DIR}/vvconfig.h)


#---------------------------------------------------------------------------------------------------
# Add virvo library
#


add_definitions(-DHAVE_CONFIG_H)


set(VIRVO_HEADERS
  private/archives.h
  private/blocking_queue.h
  private/connection.h
  private/connection_manager.h
  private/message_queue.h
  private/vvcompress.h
  private/vvcompressedvector.h
  private/vvgltools.h
  private/vvibrimage.h
  private/vvimage.h
  private/vvlog.h
  private/vvmessage.h
  private/project.h
  private/project.impl.h
  private/vvserialize.h
  private/vvtimer.h
  private/work_queue.h

  cuda/array.h
  cuda/debug.h
  cuda/graphics_resource.h
  cuda/host_device_array.h
  cuda/memory.h
  cuda/symbol.h
  cuda/texture.h
  cuda/utils.h

  fileio/exceptions.h
  fileio/nifti.h
  fileio/nrrd.h

  gl/handle.h
  gl/types.h
  gl/util.h

  math/detail/simd/matrix4.inl
  math/detail/aabb.inl
  math/detail/math.h
  math/detail/matrix4.inl
  math/detail/plane3.inl
  math/detail/rectangle.inl
  math/detail/vector.inl
  math/detail/vector2.inl
  math/detail/vector3.inl
  math/detail/vector4.inl
  math/simd/matrix.h
  math/simd/sse.h
  math/simd/intrinsics.h
  math/aabb.h
  math/forward.h
  math/intersect.h
  math/matrix.h
  math/math.h
  math/output.h
  math/plane.h
  math/ray.h
  math/rectangle.h
  math/serialization.h

  mem/allocator.h
  mem/new.h
  mem/swap.h

  texture/detail/prefilter.h
  texture/detail/sampler1d.h
  texture/detail/sampler3d.h
  texture/detail/sampler_common.h
  texture/detail/texture1d.h
  texture/detail/texture3d.h
  texture/detail/texture_common.h
  texture/forward.h
  texture/texture.h

  vvbrickrend.h
  vvbsptree.h
  vvbsptreevisitors.h
  vvcgprogram.h
  vvclipobj.h
  vvclock.h
  vvcolor.h
  vvcompiler.h
  vvcudaimg.h
  vvcudarendertarget.h
  vvcudatransfunc.h
  vvdebugmsg.h
  vvdicom.h
  vvdynlib.h
  vvexport.h
  vvfileio.h
  vvglslprogram.h
  vvibr.h
  vvibrclient.h
  vvibrserver.h
  vvimage.h
  vvimageclient.h
  vvimageserver.h
  vvinttypes.h
  vvmacros.h
  vvmulticast.h
  vvopengl.h
  vvparam.h
  vvparbrickrend.h
  vvpixelformat.h
  vvplatform.h
  vvprintgl.h
  vvpthread.h
  vvremoteclient.h
  vvremoteevents.h
  vvremoteserver.h
  vvrendercontext.h
  vvrenderer.h
  vvrendererfactory.h
  vvrendertarget.h
  vvrendervp.h
  vvrequestmanagement.h
  vvserbrickrend.h
  vvserialization.h
  vvshaderfactory.h
  vvshaderprogram.h
  vvsllist.h
  vvsocket.h
  vvsocketio.h
  vvsocketmap.h
  vvsocketmonitor.h
  vvsoftimg.h
  vvsoftpar.h
  vvsoftper.h
  vvsoftsw.h
  vvsoftvr.h
  vvstingray.h
  vvswitchrenderer.h
  vvswitchrenderer.impl.h
  vvtcpserver.h
  vvtcpsocket.h
  vvtexrend.h
  vvtextureutil.h
  vvtfwidget.h
  vvtokenizer.h
  vvtoolshed.h
  vvtoolshed.impl.h
  vvtransfunc.h
  vvudpsocket.h
  vvvecmath.h
  vvvecmath.impl.h
  vvvffile.h
  vvvideo.h
  vvvirvo.h
  vvvisitor.h
  vvvoldesc.h
)

set(VIRVO_SOURCES
  private/connection.cpp
  private/connection_manager.cpp
  private/vvcompress_jpeg.cpp
  private/vvcompress_png.cpp
  private/vvcompress_snappy.cpp
  private/vvgltools.cpp
  private/vvibrimage.cpp
  private/vvimage.cpp
  private/vvmessage.cpp

  cuda/debug.cpp
  cuda/graphics_resource.cpp
  cuda/host_device_array.cpp
  cuda/utils.cpp

  gl/handle.cpp
  gl/util.cpp

  vvbrickrend.cpp
  vvbsptree.cpp
  vvbsptreevisitors.cpp
  vvcgprogram.cpp
  vvclipobj.cpp
  vvcudaimg.cpp
  vvcudarendertarget.cpp
# vvcudarendertarget.inc
# vvcudarendertarget-null.inc
  vvcudatransfunc.cpp
  vvdynlib.cpp
  vvglslprogram.cpp
  vvibr.cpp
  vvibrclient.cpp
  vvibrserver.cpp
  vvimage.cpp
  vvimageclient.cpp
  vvimageserver.cpp
  vvmulticast.cpp
  vvparam.cpp
  vvparbrickrend.cpp
  vvpixelformat.cpp
  vvprintgl.cpp
  vvpthread.cpp
  vvremoteclient.cpp
  vvremoteserver.cpp
  vvrendercontext.cpp
  vvrenderer.cpp
  vvrendererfactory.cpp
  vvrendertarget.cpp
  vvrendervp.cpp
  vvrequestmanagement.cpp
  vvserbrickrend.cpp
  vvshaderfactory.cpp
  vvshaderprogram.cpp
  vvsocket.cpp
  vvsocketio.cpp
  vvsocketmap.cpp
  vvsocketmonitor.cpp
  vvsoftimg.cpp
  vvsoftpar.cpp
  vvsoftper.cpp
  vvsoftsw.cpp
  vvsoftvr.cpp
  vvstingray.cpp
  vvtcpserver.cpp
  vvtcpsocket.cpp
  vvtexrend.cpp
  vvtextureutil.cpp
  vvudpsocket.cpp
  vvvecmath.cpp
  vvvffile.cpp
  vvvideo.cpp
  vvvirvo.cpp
  vvvisitor.cpp
)

if(DESKVOX_USE_CUDA)
deskvox_cuda_compiles(VIRVO_CUDA_SOURCES
  vvcudatransfunc.cu
)
endif(DESKVOX_USE_CUDA)

if(DESKVOX_USE_ZEROCONF)
  set(VIRVO_HEADERS
    ${VIRVO_HEADERS}
    vvbonjour/vvbonjour.h
    vvbonjour/vvbonjourbrowser.h
    vvbonjour/vvbonjourentry.h
    vvbonjour/vvbonjoureventloop.h
    vvbonjour/vvbonjourregistrar.h
    vvbonjour/vvbonjourresolver.h
  )

  set(VIRVO_SOURCES
    ${VIRVO_SOURCES}
    vvbonjour/vvbonjour.cpp
    vvbonjour/vvbonjourbrowser.cpp
    vvbonjour/vvbonjourentry.cpp
    vvbonjour/vvbonjoureventloop.cpp
    vvbonjour/vvbonjourregistrar.cpp
    vvbonjour/vvbonjourresolver.cpp
  )
endif()

set(VIRVO_SHADERS
  ../shader/vv_ibr.fsh
  ../shader/vv_ibr.vsh
  ../shader/vv_isect_geom_only.gsh
  ../shader/vv_isect_geom_only.vsh
  ../shader/vv_isect_vert_geom_combined.gsh
  ../shader/vv_isect_vert_geom_combined.vsh
  ../shader/vv_isect_vert_only.vsh
  ../shader/vv_shader01.fsh
  ../shader/vv_shader02.fsh
  ../shader/vv_shader03.fsh
  ../shader/vv_shader04.fsh
  ../shader/vv_shader05.fsh
  ../shader/vv_shader06.fsh
  ../shader/vv_shader07.fsh
  ../shader/vv_shader08.fsh
  ../shader/vv_shader09.fsh
  ../shader/vv_shader10.fsh
  ../shader/vv_shader11.fsh
  ../shader/vv_shader12.fsh
  ../shader/vv_shader13.fsh
  ../shader/vv_shader14.fsh
  ../shader/vv_texrend.fsh
)

if(APPLE)
  set(VIRVO_WINDOWING_SYSTEM "Cocoa" CACHE STRING "Available windowing systems for render contexts (Cocoa | X11)")
elseif(WIN32)
  set(VIRVO_WINDOWING_SYSTEM "WGL" CACHE STRING "Available windowing systems for render contexts (WGL | X11)")
else()
  set(VIRVO_WINDOWING_SYSTEM "X11" CACHE STRING "Available windowing systems for render contexts (X11)")
endif()

if(APPLE)
  if(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
    add_definitions(-DUSE_COCOA)
    set(VIRVO_HEADERS ${VIRVO_HEADERS} vvcocoaglcontext.h)
    set(VIRVO_SOURCES ${VIRVO_SOURCES} vvcocoaglcontext.mm)
    find_library(COCOA_LIBRARY Cocoa)
    if(NOT COCOA_LIBRARY)
      message(FATAL_ERROR "Cocoa not found")
    endif()
    deskvox_link_libraries(${COCOA_LIBRARY})
  else(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
    if(X11_FOUND)
      ADD_DEFINITIONS(-DUSE_X11)
    endif(X11_FOUND)
  endif(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
else(APPLE)
  if(${VIRVO_WINDOWING_SYSTEM} STREQUAL "X11")
    if(X11_FOUND)
      add_definitions(-DUSE_X11)
    endif(X11_FOUND)
  endif(${VIRVO_WINDOWING_SYSTEM} STREQUAL "X11")
endif(APPLE)

if(WIN32)
  deskvox_link_libraries(ws2_32)
  deskvox_link_libraries(bcrypt)
  if(MINGW)
    deskvox_link_libraries(mswsock)
  endif()
else()
  if(BUILD_SHARED_LIBS)
    deskvox_link_libraries(dl)
  endif()
endif()

if(DESKVOX_USE_GDCM AND GDCM_FOUND)
   deskvox_link_libraries(gdcmDSED gdcmMSFF gdcmDICT gdcmCommon)
endif()

deskvox_link_libraries(virvo_fileio)

deskvox_add_library(virvo
  ${VIRVO_HEADERS}
  ${VIRVO_SOURCES}
  ${VIRVO_CUDA_SOURCES}
  ${VIRVO_SHADERS}
)

# MSVC + CUDA: link with legacy stdio library

if (MSVC_VERSION GREATER 1800 AND DESKVOX_USE_CUDA AND CUDA_FOUND)
    target_link_libraries(virvo legacy_stdio_definitions.lib)
endif()
