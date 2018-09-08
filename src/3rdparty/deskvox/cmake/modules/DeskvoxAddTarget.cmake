include(AddFileDependencies)


#---------------------------------------------------------------------------------------------------
# deskvox_link_libraries(libraries...)
#


macro(deskvox_link_libraries)
  set(__DESKVOX_LINK_LIBRARIES ${__DESKVOX_LINK_LIBRARIES} ${ARGN})
endmacro()


#---------------------------------------------------------------------------------------------------
# deskvox_cuda_compiles(outfiles, sources...)
#


function(deskvox_cuda_compiles outfiles)
  if(NOT CUDA_FOUND)
    return()
  endif()

  foreach(f ${ARGN})
    if(BUILD_SHARED_LIBS)
      cuda_compile(cuda_compile_obj ${f} SHARED)
    else()
      cuda_compile(cuda_compile_obj ${f})
    endif()
    set(out ${out} ${f} ${cuda_compile_obj})
  endforeach()

  set(${outfiles} ${out} PARENT_SCOPE)
endfunction()


#---------------------------------------------------------------------------------------------------
# __deskvox_process_sources(sources...)
#


function(__deskvox_process_sources)
  foreach(f ${ARGN})
    set(group)

    if(DESKVOX_GROUP_SOURCES_BY_TYPE)
      get_filename_component(ext ${f} EXT)

      if(ext MATCHES "\\.(h|hpp|hxx|inl|inc)")
        set(group "include")
      elseif(ext MATCHES "\\.(c|cu|cpp|cxx|mm)")
        set(group "src")
      else()
        set(group "resources")
      endif()
    endif()

    get_filename_component(path ${f} PATH)

    if(NOT path STREQUAL "")
      string(REPLACE "/" "\\" path "${path}")
      set(group "${group}\\${path}")
    endif()

    source_group("${group}" FILES ${f})

    list(APPEND out ${f})
  endforeach()

  set(__DESKVOX_PROCESSED_SOURCES ${out} PARENT_SCOPE)
endfunction()


#---------------------------------------------------------------------------------------------------
# __deskvox_set_target_postfixes(target)
#


function(__deskvox_set_target_postfixes target)
  if(BUILD_SHARED_LIBS)
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "${DESKVOX_SHARED_LIB_POSTFIX_DEBUG}")
    set_target_properties(${target} PROPERTIES RELEASE_POSTFIX "${DESKVOX_SHARED_LIB_POSTFIX_RELEASE}")
    set_target_properties(${target} PROPERTIES MINSIZEREL_POSTFIX "${DESKVOX_SHARED_LIB_POSTFIX_MINSIZEREL}")
    set_target_properties(${target} PROPERTIES RELWITHDEBINFO_POSTFIX "${DESKVOX_SHARED_LIB_POSTFIX_RELWITHDEBINFO}")
  else()
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "${DESKVOX_STATIC_LIB_POSTFIX_DEBUG}")
    set_target_properties(${target} PROPERTIES RELEASE_POSTFIX "${DESKVOX_STATIC_LIB_POSTFIX_RELEASE}")
    set_target_properties(${target} PROPERTIES MINSIZEREL_POSTFIX "${DESKVOX_STATIC_LIB_POSTFIX_MINSIZEREL}")
    set_target_properties(${target} PROPERTIES RELWITHDEBINFO_POSTFIX "${DESKVOX_STATIC_LIB_POSTFIX_RELWITHDEBINFO}")
  endif()
endfunction()


#---------------------------------------------------------------------------------------------------
# deskvox_add_library(name, sources...)
#


function(deskvox_add_library name)
  message(STATUS "Adding library " ${name} "...")

  __deskvox_process_sources(${ARGN})

  add_library(${name} ${__DESKVOX_PROCESSED_SOURCES})

  set_target_properties(${name} PROPERTIES FOLDER "Libraries")

  # Hide all symbols by default
  if(DESKVOX_COMPILER_IS_GCC_COMPATIBLE)
    get_property(COMPILE_FLAGS_OLD TARGET ${name} PROPERTY COMPILE_FLAGS)
    set_target_properties(${name} PROPERTIES "COMPILE_FLAGS" "${COMPILE_FLAGS_OLD} -fvisibility=hidden")
  endif()

  if(NOT COVISE_BUILD)
    __deskvox_set_target_postfixes(${name})
  endif()

  target_link_libraries(${name} ${__DESKVOX_LINK_LIBRARIES})


  if(COVISE_BUILD)
    install(TARGETS ${name} EXPORT covise-targets
      RUNTIME DESTINATION ${ARCHSUFFIX}/bin${_category_path}
      LIBRARY DESTINATION ${ARCHSUFFIX}/lib
      ARCHIVE DESTINATION ${ARCHSUFFIX}/lib COMPONENT modules.${category}
    )
  else(COVISE_BUILD)
    install(TARGETS ${name}
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
    )
  endif(COVISE_BUILD)
endfunction()


#---------------------------------------------------------------------------------------------------
# deskvox_add_plugin(name, sources...)
#


function(deskvox_add_plugin name)
  message(STATUS "Adding plugin " ${name} "...")

  __deskvox_process_sources(${ARGN})

  add_library(${name} ${__DESKVOX_PROCESSED_SOURCES})

  set_target_properties(${name} PROPERTIES FOLDER "Plugins")

  __deskvox_set_target_postfixes(${name})

  target_link_libraries(${name} ${__DESKVOX_LINK_LIBRARIES})

  if(COVISE_BUILD)
    install(TARGETS ${name} EXPORT covise-targets
      RUNTIME DESTINATION ${ARCHSUFFIX}/bin
      LIBRARY DESTINATION ${ARCHSUFFIX}/lib
      ARCHIVE DESTINATION ${ARCHSUFFIX}/lib
    )
  else(COVISE_BUILD)
    install(TARGETS ${name}
      RUNTIME DESTINATION plugins
      LIBRARY DESTINATION plugins
      ARCHIVE DESTINATION plugins
    )
  endif(COVISE_BUILD)
endfunction()


#---------------------------------------------------------------------------------------------------
# __deskvox_add_executable(folder, name, sources...)
#


function(__deskvox_add_executable folder name)
  message(STATUS "Adding executable: " ${name} " (" ${folder} ")...")

  __deskvox_process_sources(${ARGN})

  add_executable(${name} ${__DESKVOX_PROCESSED_SOURCES})

  set_target_properties(${name} PROPERTIES FOLDER ${folder})

  #__deskvox_set_target_postfixes(${name})

  target_link_libraries(${name} ${__DESKVOX_LINK_LIBRARIES})

  if(COVISE_BUILD)
    install(TARGETS ${name} EXPORT covise-targets
        RUNTIME DESTINATION ${ARCHSUFFIX}/bin
    )
  else(COVISE_BUILD)
    install(TARGETS ${name} RUNTIME DESTINATION bin)
  endif(COVISE_BUILD)
endfunction()


#---------------------------------------------------------------------------------------------------
# deskvox_add_tool(name, sources...)
#


function(deskvox_add_tool name)
  __deskvox_add_executable("Tools" ${name} ${ARGN})
endfunction()


#---------------------------------------------------------------------------------------------------
# deskvox_add_test(name, sources...)
#


function(deskvox_add_test name)
  __deskvox_add_executable("Tests" ${name} ${ARGN})
endfunction()
