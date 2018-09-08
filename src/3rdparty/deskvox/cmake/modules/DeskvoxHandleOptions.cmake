# This CMake module is responsible for interpreting the user defined DESKVOX_* options and
# executing the appropriate CMake commands to realize the users' selections.


if(DESKVOX_USE_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()


if(NOT BUILD_SHARED_LIBS)
  add_definitions(-DNODLL)
endif()


if(WIN32)
  add_definitions(-D_UNICODE -DUNICODE)
endif()


if(MSVC)

  add_definitions(
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_SECURE_NO_WARNINGS
    -D_CRT_NONSTDC_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_WARNINGS
    -D_SCL_SECURE_NO_DEPRECATE
    -D_SCL_SECURE_NO_WARNINGS
  )

  # Disable warning:
  #
  # C4251: 'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'
  # C4275: non-DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'
  # C4481: nonstandard extension used: override specifier 'keyword'
  # C4503: 'identifier' : decorated name length exceeded, name was truncated
  # C4512: 'class' : assignment operator could not be generated
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4481 /wd4503 /wd4512")

  # Promote to level 1 warnings:
  #
  # C4062: enumerator in switch of enum is not handled
  # C4146:unary minus operator applied to unsigned type, result still unsigned
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w14062 /w14146")

  # Promote to errors:
  #
  # C4238: Don't take address of temporaries
  # C4239: Don't bind temporaries to non-const references (Stephan's "Evil Extension")
  # C4288: For-loop scoping (this is the default)
  # C4346: Require "typename" where the standard requires it
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /we4238 /we4239 /we4288 /we4346")

  if(DESKVOX_ENABLE_WARNINGS)
    deskvox_replace_compiler_option(CMAKE_CXX_FLAGS "/W3" "/W4")
    if(DESKVOX_ENABLE_PEDANTIC)
      deskvox_replace_compiler_option(CMAKE_CXX_FLAGS "/W4" "/Wall")
    endif()
  endif()
  if(DESKVOX_ENABLE_WERROR)
    add_definitions(/WX)
  endif()

elseif(DESKVOX_COMPILER_IS_GCC_COMPATIBLE)

  add_definitions(-Wmissing-braces)
  add_definitions(-Wsign-compare)
  add_definitions(-Wwrite-strings)
  add_definitions(-Woverloaded-virtual)

  # Disable -Wlong-long...
  add_definitions(-Wno-long-long)

  if(DESKVOX_ENABLE_WARNINGS)
    add_definitions(-Wall -Wextra)
    if(DESKVOX_ENABLE_PEDANTIC)
      add_definitions(-pedantic)
    endif()
  endif()
  if(DESKVOX_ENABLE_WERROR)
    add_definitions(-Werror)
  endif()

endif()
