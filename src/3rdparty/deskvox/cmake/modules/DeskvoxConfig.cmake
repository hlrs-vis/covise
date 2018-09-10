include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckCXXSourceCompiles)


if(CMAKE_COMPILER_IS_GNUCXX)
  set(DESKVOX_COMPILER_IS_GCC_COMPATIBLE ON)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(DESKVOX_COMPILER_IS_GCC_COMPATIBLE ON)
endif()

#if(DESKVOX_COMPILER_IS_GCC_COMPATIBLE)
#    add_definitions(-std=c++0x)
#endif()


#---------------------------------------------------------------------------------------------------
# deskvox_replace_compiler_option(var, old, new)
#
# Replaces a compiler option or switch <old> in <var> by <new>
# If <old> is not in <var>, appends <new> to <var>
# Example:
#
#	deskvox_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O2")
#


function(deskvox_replace_compiler_option var old new)
  # If the option already is on the variable, don't add it:
  if("${${var}}" MATCHES "(^| )${new}($| )")
    set(n "")
  else()
    set(n "${new}")
  endif()
  if("${${var}}" MATCHES "(^| )${old}($| )")
    string(REGEX REPLACE "(^| )${old}($| )" " ${n} " ${var} "${${var}}")
  else()
    set(${var} "${${var}} ${n}")
  endif()
  set(${var} "${${var}}" PARENT_SCOPE)
endfunction()


#---------------------------------------------------------------------------------------------------
# include checks
#

#check_include_file(dlfcn.h HAVE_DLFCN_H)
#check_include_file(execinfo.h HAVE_EXECINFO_H)
#check_include_file(stdint.h HAVE_STDINT_H)
#check_include_file(pthread.h HAVE_PTHREAD_H)


#---------------------------------------------------------------------------------------------------
# library checks
#


#check_library_exists(pthread pthread_create "" HAVE_LIBPTHREAD)


#---------------------------------------------------------------------------------------------------
# function checks
#


#---------------------------------------------------------------------------------------------------
# type checks
#


function(deskvox_check_type_exists type result)
  check_cxx_source_compiles(
    "#include <stdint.h>
    ${type} var;
    int main() { return 0; }" ${result})
endfunction()

deskvox_check_type_exists("long long" VV_HAVE_LLONG)
deskvox_check_type_exists("unsigned long long" VV_HAVE_ULLONG)
