USING(Revit)
cmake_minimum_required(VERSION 3.0)
PROJECT(RevitPlugin CSharp)
include(CSharpUtilities)

add_library(OpenFOAMInterface SHARED
BIM/AssemblyInfo.cs
BIM/DataGenerator.cs
BIM/OpenFOAMDialogManager.cs
BIM/OpenFOAMSimulateCommand.cs
BIM/OpenFOAMInterfaceApp.cs
BIM/OpenFOAMInterfaceResource.Designer.cs
BIM/OpenFOAMInterfaceResource.resx
BIM/SaveData.cs
BIM/Settings.cs
BIM/OpenFOAMCancelForm.cs
BIM/OpenFOAMCancelForm.Designer.cs
BIM/OpenFOAMCancelForm.resx
BIM/Properties/Resources.Designer.cs
BIM/Properties/Resources.resx
BIM/OpenFOAMUI/OpenFOAMTextBoxForm.Designer.cs
BIM/OpenFOAMUI/OpenFOAMTextBoxForm.cs
BIM/OpenFOAMUI/OpenFOAMTextBoxForm.resx
BIM/OpenFOAM/Alphat.cs
BIM/OpenFOAM/BlockMeshDict.cs
BIM/OpenFOAM/ControlDict.cs
BIM/OpenFOAM/DecomposeParDict.cs
BIM/OpenFOAM/Epsilon.cs
BIM/OpenFOAM/FoamDict.cs
BIM/OpenFOAM/FoamFile.cs
BIM/OpenFOAM/FoamParameter.cs
BIM/OpenFOAM/FvSchemes.cs
BIM/OpenFOAM/FvSolution.cs
BIM/OpenFOAM/G.cs
BIM/OpenFOAM/K.cs
BIM/OpenFOAM/MeshDict.cs
BIM/OpenFOAM/Nut.cs
BIM/OpenFOAM/OpenFOAMCalculator.cs
BIM/OpenFOAM/P.cs
BIM/OpenFOAM/P_rgh.cs
BIM/OpenFOAM/RunManager.cs
BIM/OpenFOAM/SnappyHexMeshDict.cs
BIM/OpenFOAM/SurfaceFeatureExtract.cs
BIM/OpenFOAM/T.cs
BIM/OpenFOAM/TransportProperties.cs
BIM/OpenFOAM/TurbulenceProperties.cs
BIM/OpenFOAM/U.cs
BIM/OpenFOAM/Version.cs
BIM/Resources/OpenFOAMInterface.addin
BIM/Resources/openfoaminterface.png
)

set_property(SOURCE BIM/Resources/OpenFoamInterfaceHelp.html PROPERTY VS_TOOL_OVERRIDE "Resource")
set_property(SOURCE BIM/Resources/DropDownTreeView.dll PROPERTY VS_TOOL_OVERRIDE "Resource")
set_property(SOURCE BIM/Resources/OpenFOAMInterface.addin PROPERTY VS_TOOL_OVERRIDE "Resource")
set_property(SOURCE BIM/Resources/PackageContents.xml PROPERTY VS_TOOL_OVERRIDE "Resource")
set_property(SOURCE BIM/Resources/openfoaminterface.png PROPERTY VS_TOOL_OVERRIDE "Resource")

# Set the C# language version (defaults to 3.0).
set(CMAKE_CSharp_FLAGS "/langversion:latest")
set_target_properties(OpenFOAMInterface PROPERTIES FOLDER "Plugins/RevitPlugins")

SET(BINLIB_SUFFIX "lib")
SET(MYPATH_POSTFIX )
set_target_properties(OpenFOAMInterface PROPERTIES OUTPUT_NAME OpenFOAMInterface$<$<CONFIG:Debug>:_d>)
set_target_properties(OpenFOAMInterface PROPERTIES RELEASE_OUTPUT_NAME OpenFOAMInterface)
set_target_properties(OpenFOAMInterface PROPERTIES DEBUG_OUTPUT_NAME OpenFOAMInterface_d)

IF(CMAKE_CONFIGURATION_TYPES)
      # generator supports configuration types
      FOREACH(conf_type ${CMAKE_CONFIGURATION_TYPES})
        STRING(TOUPPER "${conf_type}" upper_conf_type_str)
        IF(upper_conf_type_str STREQUAL "DEBUG")
            SET_TARGET_PROPERTIES(OpenFOAMInterface PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
            SET_TARGET_PROPERTIES(OpenFOAMInterface PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
            SET_TARGET_PROPERTIES(OpenFOAMInterface PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
        ELSE(upper_conf_type_str STREQUAL "DEBUG")
            SET_TARGET_PROPERTIES(OpenFOAMInterface PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}opt/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
            SET_TARGET_PROPERTIES(OpenFOAMInterface PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}opt/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
            SET_TARGET_PROPERTIES(OpenFOAMInterface PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${upper_conf_type_str} "${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}opt/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
        ENDIF(upper_conf_type_str STREQUAL "DEBUG")
      ENDFOREACH(conf_type)
    ELSE(CMAKE_CONFIGURATION_TYPES)
      # no configuration types - probably makefile generator
      STRING(TOUPPER "${CMAKE_BUILD_TYPE}" upper_build_type_str)
      SET_TARGET_PROPERTIES(OpenFOAMInterface PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${upper_build_type_str} "${COVISE_DESTDIR}/${ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
      SET_TARGET_PROPERTIES(OpenFOAMInterface PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${upper_build_type_str} "${COVISE_DESTDIR}/${ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
      SET_TARGET_PROPERTIES(OpenFOAMInterface PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${upper_build_type_str} "${COVISE_DESTDIR}/${ARCHSUFFIX}/${BINLIB_SUFFIX}${MYPATH_POSTFIX}")
    ENDIF(CMAKE_CONFIGURATION_TYPES)

csharp_set_designer_cs_properties(
BIM/OpenFOAMInterfaceResource.Designer.cs
BIM/OpenFOAMInterfaceResource.resx
BIM/Properties/Resources.resx
)
csharp_set_windows_forms_properties(
BIM/AssemblyInfo.cs
BIM/OpenFOAMCancelForm.cs
BIM/OpenFOAMCancelForm.Designer.cs
BIM/OpenFOAMCancelForm.resx
BIM/OpenFOAMUI/OpenFOAMTextBoxForm.cs
BIM/OpenFOAMUI/OpenFOAMTextBoxForm.Designer.cs
BIM/OpenFOAMUI/OpenFOAMTextBoxForm.resx
)
set_property(TARGET OpenFOAMInterface PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v4.8")
#set_property(TARGET OpenFOAMInterface PROPERTY WIN32_EXECUTABLE TRUE)
set_property(TARGET OpenFOAMInterface PROPERTY VS_GLOBAL_ROOTNAMESPACE "OpenFOAMInterface")

set_property(TARGET OpenFOAMInterface PROPERTY VS_DOTNET_REFERENCES
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