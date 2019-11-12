USING(Revit)
PROJECT(RevitPlugin CSharp)
include(CSharpUtilities)

add_library(OpenCOVERPlugin SHARED
ByteSwap.cs
ConnectionDialog.cs
ConnectionDialog.Designer.cs
ConnectionDialog.resx
COVER.cs
COVERToolbar.cs
ExportContext.cs
MessageBuffer.cs
Properties/AssemblyInfo.cs
)
# Set the C# language version (defaults to 3.0).
set(CMAKE_CSharp_FLAGS "/langversion:latest")

SET(BINLIB_SUFFIX "lib")
SET(MYPATH_POSTFIX )

set_target_properties(OpenCOVERPlugin PROPERTIES FOLDER "Plugins/RevitPlugins")
set_target_properties(OpenCOVERPlugin PROPERTIES OUTPUT_NAME OpenCOVERPlugin$<$<CONFIG:Debug>:_d>)

IF(CMAKE_CONFIGURATION_TYPES)
      # generator supports configuration types
      FOREACH(conf_type ${CMAKE_CONFIGURATION_TYPES})
        STRING(TOUPPER "${conf_type}" upper_conf_type_str)
        IF(upper_conf_type_str STREQUAL "DEBUG")
            SET_TARGET_PROPERTIES(OpenCOVERPlugin PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
            SET_TARGET_PROPERTIES(OpenCOVERPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
            SET_TARGET_PROPERTIES(OpenCOVERPlugin PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
        ELSE(upper_conf_type_str STREQUAL "DEBUG")
            SET_TARGET_PROPERTIES(OpenCOVERPlugin PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}opt/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
            SET_TARGET_PROPERTIES(OpenCOVERPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}opt/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
            SET_TARGET_PROPERTIES(OpenCOVERPlugin PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}opt/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
        ENDIF(upper_conf_type_str STREQUAL "DEBUG")
      ENDFOREACH(conf_type)
    ELSE(CMAKE_CONFIGURATION_TYPES)
      # no configuration types - probably makefile generator
      STRING(TOUPPER "${CMAKE_BUILD_TYPE}" upper_build_type_str)
      SET_TARGET_PROPERTIES(OpenCOVERPlugin PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${upper_build_type_str} "${COVISE_DESTDIR}/${ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
      SET_TARGET_PROPERTIES(OpenCOVERPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${upper_build_type_str} "${COVISE_DESTDIR}/${ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
      SET_TARGET_PROPERTIES(OpenCOVERPlugin PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${upper_build_type_str} "${COVISE_DESTDIR}/${ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
    ENDIF(CMAKE_CONFIGURATION_TYPES)

csharp_set_designer_cs_properties(
Properties/AssemblyInfo.cs
ConnectionDialog.cs
ConnectionDialog.Designer.cs
ConnectionDialog.resx
)
set_property(TARGET OpenCOVERPlugin PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v4.7.1")
#set_property(TARGET OpenCOVERPlugin PROPERTY WIN32_EXECUTABLE TRUE)

set_property(TARGET OpenCOVERPlugin PROPERTY VS_DOTNET_REFERENCES
    ${EXTRA_REFS}
    "PresentationCore"
    "System"
    "System.Core"
    "System.Data"
    "System.Drawing"
    "System.Windows.Forms"
    "System.Windows.Presentation"
    "System.Xaml"
    "System.Xml"
    "WindowsBase")