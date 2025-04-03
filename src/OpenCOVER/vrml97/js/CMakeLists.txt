# @file
# 
# CMakeLists.txt for kernel - vrml - js
#
# @author Blasius Czink

ADD_DEFINITIONS(-DHAVE_CONFIG_H -DJS_USE_ONLY_NSPR_LOCKS -DJSFILE -DEXPORT_JS_API)

INCLUDE(CheckTypeSize)
CHECK_TYPE_SIZE("long long" LONG_LONG)
IF(HAVE_LONG_LONG)
  ADD_DEFINITIONS(-DHAVE_LONG_LONG)
ENDIF()

IF(UNIX)
  SET(JS_BUILD_TYPE STATIC)
  ADD_DEFINITIONS(-DXP_UNIX)
ELSE()  
  SET(JS_BUILD_TYPE SHARED)
ENDIF()

INCLUDE_DIRECTORIES(
  ..
)

SET(JS_SOURCES
  jsapi.c jsarena.c jsarray.c jsatom.c jsbool.c jscntxt.c jsdate.c jsdbgapi.c
  jsdhash.c jsdtoa.c jsemit.c jsexn.c jsfile.c jsfun.c jsgc.c jshash.c
  jsinterp.c jslock.c jslog2.c jslong.c jsmath.c jsnum.c jsobj.c jsopcode.c
  jsparse.c jsprf.c jsregexp.c jsscan.c jsscope.c jsscript.c jsstr.c jsutil.c
  jsxdrapi.c jsxml.c prmjtime.c
  fdlibm/e_acos.c fdlibm/e_acosh.c fdlibm/e_asin.c fdlibm/e_atan2.c
  fdlibm/e_atanh.c fdlibm/e_cosh.c fdlibm/e_exp.c fdlibm/e_fmod.c
  fdlibm/e_gamma.c fdlibm/e_gamma_r.c fdlibm/e_hypot.c fdlibm/e_j0.c
  fdlibm/e_j1.c fdlibm/e_jn.c fdlibm/e_lgamma.c fdlibm/e_lgamma_r.c
  fdlibm/e_log10.c fdlibm/e_log.c fdlibm/e_pow.c fdlibm/e_remainder.c
  fdlibm/e_rem_pio2.c fdlibm/e_scalb.c fdlibm/e_sinh.c fdlibm/e_sqrt.c
  fdlibm/k_cos.c fdlibm/k_rem_pio2.c fdlibm/k_sin.c fdlibm/k_standard.c
  fdlibm/k_tan.c fdlibm/s_asinh.c fdlibm/s_atan.c fdlibm/s_cbrt.c
  fdlibm/s_ceil.c fdlibm/s_copysign.c fdlibm/s_cos.c fdlibm/s_erf.c
  fdlibm/s_expm1.c fdlibm/s_fabs.c fdlibm/s_finite.c fdlibm/s_floor.c
  fdlibm/s_frexp.c fdlibm/s_ilogb.c fdlibm/s_isnan.c fdlibm/s_ldexp.c
  fdlibm/s_lib_version.c fdlibm/s_log1p.c fdlibm/s_logb.c fdlibm/s_matherr.c
  fdlibm/s_modf.c fdlibm/s_nextafter.c fdlibm/s_rint.c fdlibm/s_scalbn.c
  fdlibm/s_signgam.c fdlibm/s_significand.c fdlibm/s_sin.c fdlibm/s_tan.c
  fdlibm/s_tanh.c fdlibm/w_acos.c fdlibm/w_acosh.c fdlibm/w_asin.c
  fdlibm/w_atan2.c fdlibm/w_atanh.c fdlibm/w_cosh.c fdlibm/w_exp.c
  fdlibm/w_fmod.c fdlibm/w_gamma.c fdlibm/w_gamma_r.c fdlibm/w_hypot.c
  fdlibm/w_j0.c fdlibm/w_j1.c fdlibm/w_jn.c fdlibm/w_lgamma.c
  fdlibm/w_lgamma_r.c fdlibm/w_log10.c fdlibm/w_log.c fdlibm/w_pow.c
  fdlibm/w_remainder.c fdlibm/w_scalb.c fdlibm/w_sinh.c fdlibm/w_sqrt.c
)

SET(JS_HEADERS
  jsapi.h jsarena.h jsarray.h jsatom.h jsbit.h jsbool.h jsclist.h
  jscntxt.h jscompat.h jsconfig.h jsdate.h jsdbgapi.h jsdhash.h jsdtoa.h
  jsemit.h jsexn.h jsfile.h jsfun.h jsgc.h jshash.h jsinterp.h jslibmath.h
  jslock.h jslong.h jsmath.h jsnum.h jsobj.h jsopcode.h jsosdep.h jsotypes.h
  jsparse.h jsprf.h jsprvtd.h jspubtd.h jsregexp.h jsscan.h jsscope.h
  jsscript.h jsstddef.h jsstr.h jstypes.h jsutil.h jsxdrapi.h jsxml.h
  prmjtime.h resource.h
)

IF(UNIX)
  ADD_EXECUTABLE(jsautocfg jscpucfg.c)
  # add the custom command to generate jsautocfg.h
  ADD_CUSTOM_COMMAND(
    OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/jsautocfg.h"
    COMMAND jsautocfg > ${CMAKE_CURRENT_BINARY_DIR}/jsautocfg.h
    DEPENDS jsautocfg
  )
  LIST(APPEND JS_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/jsautocfg.h")
ENDIF(UNIX)


ADD_COVISE_LIBRARY(coJS ${JS_BUILD_TYPE} ${JS_SOURCES} ${JS_HEADERS})

IF(WIN32 AND NOT MINGW)
   ADD_COVISE_LINK_FLAGS(coJS "/OPT:NOICF")
ENDIF()
IF(UNIX)
  ADD_COVISE_COMPILE_FLAGS(coJS "-fPIC;-fno-strict-aliasing")
ENDIF() 
COVISE_WNOERROR(coJS)

# TARGET_LINK_LIBRARIES(coJS ___)

COVISE_INSTALL_TARGET(coJS)
