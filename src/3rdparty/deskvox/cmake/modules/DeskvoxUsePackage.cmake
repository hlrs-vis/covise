if(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} GREATER 2.8.3)
  include(CMakeParseArguments)
else()
  include(compatibility/CMakeParseArguments)
endif()


#---------------------------------------------------------------------------------------------------
# deskvox_use_package(name [INCDIRS include_directories...] [LIBS link_libraries...])
#


function(deskvox_use_package name)
  string(TOUPPER ${name} upper_name)

  if(NOT ${name}_FOUND AND NOT ${upper_name}_FOUND)
    return()
  endif()

  set(__DESKVOX_USED_PACKAGES ${__DESKVOX_USED_PACKAGES} ${upper_name} PARENT_SCOPE)

  set(singleArgNames FOUND)
  set(multiArgNames INCDIRS LIBS)

  cmake_parse_arguments(dup "" "${singleArgNames}" "${multiArgNames}" ${ARGN})

  #
  # If no include directories are specified, check for
  # existing cmake variables in the following order:
  #
  #   name_INCLUDE_DIR,  NAME_INCLUDE_DIR,  name_INCLUDE_DIRS,  NAME_INCLUDE_DIRS
  #
  if(NOT dup_INCDIRS)
    if(${name}_INCLUDE_DIR)
      set(dup_INCDIRS ${${name}_INCLUDE_DIR})
    elseif(${upper_name}_INCLUDE_DIR)
      set(dup_INCDIRS ${${upper_name}_INCLUDE_DIR})
    elseif(${name}_INCLUDE_DIRS)
      set(dup_INCDIRS ${${name}_INCLUDE_DIRS})
    elseif(${upper_name}_INCLUDE_DIRS)
      set(dup_INCDIRS ${${upper_name}_INCLUDE_DIRS})
    endif()
  endif()

  include_directories(SYSTEM ${dup_INCDIRS})

  #
  # If no link libraries are specified, check for existing cmake
  # variables in the following order:
  #
  #   name_LIBRARIES,  NAME_LIBRARIES,  name_LIBRARY,  NAME_LIBRARY
  #
  if(NOT dup_LIBS)
    if(${name}_LIBRARIES)
      set(dup_LIBS ${${name}_LIBRARIES})
    elseif(${upper_name}_LIBRARIES)
      set(dup_LIBS ${${upper_name}_LIBRARIES})
    elseif(${name}_LIBRARY)
      set(dup_LIBS ${${name}_LIBRARY})
    elseif(${upper_name}_LIBRARY)
      set(dup_LIBS ${${upper_name}_LIBRARY})
    endif()
  endif()

  set(__DESKVOX_LINK_LIBRARIES ${__DESKVOX_LINK_LIBRARIES} ${dup_LIBS} PARENT_SCOPE)

  if(${name}_PACKAGE_DEFINITIONS)
    add_definitions(${${name}_PACKAGE_DEFINITIONS})
  endif()
  if(${name}_PACKAGE_C_FLAGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${${name}_PACKAGE_C_FLAGS}" PARENT_SCOPE)
  endif()
  if(${name}_PACKAGE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${${name}_PACKAGE_CXX_FLAGS}" PARENT_SCOPE)
  endif()
endfunction()
