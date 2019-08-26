USING(Surface)
PROJECT(Sur40COVER CSharp)
include(CSharpUtilities)

SET(SOURCES
Program.cs
Properties/AssemblyInfo.cs
#Resources/icon.png
#Resources/iconPreview.png
)
add_covise_module(Renderer Sur40COVER )

csharp_set_designer_cs_properties(
Properties/AssemblyInfo.cs
ConnectionDialog.cs
ConnectionDialog.Designer.cs
ConnectionDialog.resx
)
set_property(TARGET Sur40COVER PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v4.7.1")
#set_property(TARGET Sur40COVER PROPERTY WIN32_EXECUTABLE TRUE)

TARGET_LINK_LIBRARIES(Sur40COVER ManagedOpenCOVER) 

#set_source_files_properties(Resources/icon.png Resources/iconPreview.png PROPERTIES LANGUAGE RC)

set_property(TARGET Sur40COVER PROPERTY VS_DOTNET_REFERENCES
    "PresentationCore"
	${EXTRA_REFS}
    "System"
    "System.Core"
    "System.Data"
    "System.Drawing"
    "System.Windows.Forms"
    "System.Xml")