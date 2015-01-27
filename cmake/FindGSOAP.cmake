#
# Try to find gSOAP
#
# GSOAP_FOUND         = gSOAP was found
# GSOAP_INCLUDE_DIR   = directory with stdsoap2.h
# GSOAP_INCLUDE_DIRS  = same as above (not chached)
# GSOAP_SOAPCPP2      = absolute path of soapcpp2
# GSOAP_WSDL2H        = absolute path of wsdl2h
#
# Macros:
#
# GSOAP_TARGET(gsoap_input gsoap_output)
#
#   - Calls wsdl2h and soapcpp2
#     In case you specify a .wsdl file for "gsoap_input" wsdl2h is called first
#     then its output is passed to soapcpp2. Otherwise (in case of a header file)
#     only soapcpp2 is called. Namespace and servicename are parsed from the headerfile
#     which is given to soapcpp2.
#
#   Variables defined by this macro:
#     ${gsoap_output}_STUBS   contains the gsoap stubs
#     ${gsoap_output}_CLIENT  conatins the client sources
#     ${gsoap_output}_SERVER  contains the server sources
#     ${gsoap_output}_HEADER  contains the header passed to soapcpp2
#   
# @author Blasius Czink

IF(GSOAP_INCLUDE_DIR)
  SET(GSOAP_FIND_QUIETLY TRUE)
ENDIF(GSOAP_INCLUDE_DIR)

IF(WIN32)
  SET(GSOAP_PATH_BIN_EXT win32)
ELSE(WIN32)
  IF(APPLE)
    SET(GSOAP_PATH_BIN_EXT macosx)
  ELSE(APPLE)
    SET(GSOAP_PATH_BIN_EXT linux386)
  ENDIF(APPLE)
ENDIF(WIN32)

FIND_PATH(GSOAP_INCLUDE_DIR stdsoap2.h
  $ENV{GSOAP_HOME}
  $ENV{EXTERNLIBS}/gsoap/gsoap
  /usr/include
  /usr/local/include
  /opt/include
  /opt/local/include
  NO_DEFAULT_PATH
  DOC "gSOAP - directory where stdsoap2.h resides"
)

FIND_PATH(GSOAP_STDINC_DIR soap12.h
  $ENV{GSOAP_HOME}/import
  $ENV{EXTERNLIBS}/gsoap/gsoap/import
  /usr/include
  /usr/local/include
  /opt/include
  /opt/local/include
  /usr/share/gsoap/import
  NO_DEFAULT_PATH
  DOC "gSOAP - directory where soap12.h resides"
)

FIND_PROGRAM(GSOAP_SOAPCPP2 soapcpp2
  PATHS
  $ENV{GSOAP_HOME}/bin/${GSOAP_PATH_BIN_EXT}
  $ENV{EXTERNLIBS}/gsoap/gsoap/bin/${GSOAP_PATH_BIN_EXT}
  /usr/bin
  /usr/local/bin
  /opt/bin
  /opt/local/bin
  NO_DEFAULT_PATH
  DOC "gSOAP - soapcpp2"
)

FIND_PROGRAM(GSOAP_WSDL2H wsdl2h
  PATHS
  $ENV{GSOAP_HOME}/bin/${GSOAP_PATH_BIN_EXT}
  $ENV{EXTERNLIBS}/gsoap/gsoap/bin/${GSOAP_PATH_BIN_EXT}
  /usr/bin
  /usr/local/bin
  /opt/bin
  /opt/local/bin
  NO_DEFAULT_PATH
  DOC "gSOAP - wsdl2h"
)

MACRO(GSOAP_SOAPCPP2_TARGET soapcpp2_input soapcpp2_output)
  SET(inp ${soapcpp2_input})
  GET_FILENAME_COMPONENT(inp_base_name "${inp}" NAME_WE)
  GET_FILENAME_COMPONENT(inp_path "${inp}" ABSOLUTE)
  GET_FILENAME_COMPONENT(inp_path "${inp_path}" PATH)
  FILE(STRINGS "${inp}" matched_lines REGEX "//gsoap(.*)service name:(.*)")
  #message("matched_lines = ${matched_lines}")
  IF(matched_lines STREQUAL "")
    # probably an empty "env.h"
    # don't make people nervous in case of empty env.h
    IF(NOT inp_base_name STREQUAL "env")
      message("[GSOAP] Could not extract namespace and servicename from given headerfile (${inp})")
    ENDIF(NOT inp_base_name STREQUAL "env")
    SET(my_namespace "${inp_base_name}")
    SET(my_servicename "")
  ELSE(matched_lines STREQUAL "")
    # parse namespace and servicename from headerfile
    STRING(REGEX MATCH "//gsoap(.*)service name:(.*)" my_namespace_and_servicename "${matched_lines}")
    STRING(STRIP "${CMAKE_MATCH_1}" my_namespace)
    STRING(STRIP "${CMAKE_MATCH_2}" my_servicename)
  ENDIF(matched_lines STREQUAL "")
  # message("[GSOAP][${soapcpp2_output}] namespace = ${my_namespace}  servicename = ${my_servicename}")
  SET(${soapcpp2_output}_STUBS
    "${CMAKE_CURRENT_BINARY_DIR}/ws${my_namespace}Stub.h"
    "${CMAKE_CURRENT_BINARY_DIR}/ws${my_namespace}H.h"
    "${CMAKE_CURRENT_BINARY_DIR}/ws${my_namespace}C.cpp"
  )
  SET(${soapcpp2_output}_CLIENT
    "${CMAKE_CURRENT_BINARY_DIR}/${my_namespace}${my_servicename}Proxy.h"
    "${CMAKE_CURRENT_BINARY_DIR}/${my_namespace}${my_servicename}Proxy.cpp"
  )
  SET(${soapcpp2_output}_SERVER
    "${CMAKE_CURRENT_BINARY_DIR}/${my_namespace}${my_servicename}Service.h"
    "${CMAKE_CURRENT_BINARY_DIR}/${my_namespace}${my_servicename}Service.cpp"
  )
  #message("${soapcpp2_output}_OUTPUTS = ${${soapcpp2_output}_OUTPUTS}")
  # the following variable is only used for cleanup
  IF(my_servicename STREQUAL "")
  SET(${soapcpp2_output}_OUTPUTS ${${soapcpp2_output}_STUBS})
  ELSE(my_servicename STREQUAL "")
  SET(${soapcpp2_output}_STUBS
    "${CMAKE_CURRENT_BINARY_DIR}/${my_namespace}Stub.h"
    "${CMAKE_CURRENT_BINARY_DIR}/${my_namespace}H.h"
    "${CMAKE_CURRENT_BINARY_DIR}/${my_namespace}C.cpp"
  )
  SET(${soapcpp2_output}_OUTPUTS ${${soapcpp2_output}_STUBS} ${${soapcpp2_output}_CLIENT} ${${soapcpp2_output}_SERVER})
  SET(${soapcpp2_output}_XML_OUTPUTS "${CMAKE_CURRENT_BINARY_DIR}/${my_namespace}.xsd")
  ENDIF(my_servicename STREQUAL "")
  IF(NOT my_servicename STREQUAL "")
    LIST(APPEND ${soapcpp2_output}_XML_OUTPUTS "${CMAKE_CURRENT_BINARY_DIR}/${my_namespace}.nsmap" "${CMAKE_CURRENT_BINARY_DIR}/${my_servicename}.wsdl")
  ENDIF(NOT my_servicename STREQUAL "")
  #message("${soapcpp2_output}_XML_OUTPUTS = ${${soapcpp2_output}_XML_OUTPUTS}")
  ADD_CUSTOM_COMMAND(OUTPUT ${${soapcpp2_output}_OUTPUTS} ${${soapcpp2_output}_XML_OUTPUTS}
      COMMAND ${GSOAP_SOAPCPP2}
      ARGS -i -x -d "${CMAKE_CURRENT_BINARY_DIR}" -npws${my_namespace} -I "${GSOAP_STDINC_DIR}" -I "${inp_path}" ${inp}
      DEPENDS ${inp}
      COMMENT "[GSOAP][${soapcpp2_output}] soapcpp2 -i -x -d ${CMAKE_CURRENT_BINARY_DIR} -npws${my_namespace} -I ${GSOAP_STDINC_DIR} -I ${inp_path} ${inp}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  SET_SOURCE_FILES_PROPERTIES(${${soapcpp2_output}_OUTPUTS} PROPERTIES GENERATED TRUE)
  # SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${${soapcpp2_output}_OUTPUTS};${${soapcpp2_output}_XML_OUTPUTS}")
  
  # define target variables
  SET(${soapcpp2_output}_DEFINED TRUE)
  SET(${soapcpp2_output}_INPUT ${inp})
  SET(${soapcpp2_output}_COMPILE_FLAGS "-i -x -d ${CMAKE_CURRENT_BINARY_DIR} -npws${my_namespace} -I ${GSOAP_STDINC_DIR} -I ${inp_path} ${inp}")

  #message("[GSOAP][${soapcpp2_output}] soapcpp2 -i -x -d ${CMAKE_CURRENT_BINARY_DIR} -npws${my_namespace} -I ${GSOAP_STDINC_DIR} -I ${inp_path} ${inp}")
ENDMACRO(GSOAP_SOAPCPP2_TARGET)

MACRO(GSOAP_TARGET gsoap_input gsoap_output)
  GET_FILENAME_COMPONENT(inp_ext "${gsoap_input}" EXT)
  STRING(TOLOWER "${inp_ext}" inp_ext)
  IF(inp_ext STREQUAL ".wsdl")
    # message("got wsdl input")
    GET_FILENAME_COMPONENT(base_name "${gsoap_input}" NAME_WE)
    GET_FILENAME_COMPONENT(inp_path "${inp}" ABSOLUTE)
    GET_FILENAME_COMPONENT(inp_path "${inp_path}" PATH)
    SET(wsdl_inp "${gsoap_input}")
    SET(wsdl_out "${CMAKE_CURRENT_BINARY_DIR}/ws_${base_name}.h")
    ADD_CUSTOM_COMMAND(OUTPUT ${wsdl_out}
        COMMAND ${GSOAP_WSDL2H}
        ARGS -f -p -I ${GSOAP_STDINC_DIR} -o ${wsdl_out} ${wsdl_inp}
        DEPENDS ${wsdl_inp}
        COMMENT "[GSOAP][${gsoap_output}] wsdl2h -f -p -I ${GSOAP_STDINC_DIR} -I ${inp_path} -o ${wsdl_out} ${wsdl_inp}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    #message("[GSOAP][${gsoap_output}] wsdl2h -f -p -I ${GSOAP_STDINC_DIR} -I ${inp_path} -o ${wsdl_out} ${wsdl_inp}")
    SET(${gsoap_output}_HEADER ${wsdl_out})
    #message("${gsoap_output}_HEADER = ${${gsoap_output}_HEADER}")
    GSOAP_SOAPCPP2_TARGET(${wsdl_out} ${gsoap_output})
    
  ELSE(inp_ext STREQUAL ".wsdl")
    # message("got header input")
    SET(${gsoap_output}_HEADER ${gsoap_input})
    # message("${gsoap_output}_HEADER = ${${gsoap_output}_HEADER}")
    # just pass to soapcpp2
    GSOAP_SOAPCPP2_TARGET(${gsoap_input} ${gsoap_output})
  ENDIF(inp_ext STREQUAL ".wsdl")
ENDMACRO(GSOAP_TARGET)

IF(GSOAP_INCLUDE_DIR)
  SET(GSOAP_INCLUDE_DIRS "${GSOAP_INCLUDE_DIR}" "${GSOAP_STDINC_DIR}")
ENDIF(GSOAP_INCLUDE_DIR)

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(GSOAP DEFAULT_MSG GSOAP_INCLUDE_DIR GSOAP_STDINC_DIR GSOAP_SOAPCPP2 GSOAP_WSDL2H)
MARK_AS_ADVANCED(GSOAP_INCLUDE_DIR GSOAP_STDINC_DIR GSOAP_SOAPCPP2 GSOAP_WSDL2H)
