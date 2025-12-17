#define COVISEDIR GetEnv("COVISEDIR")
#define REVIT_VERSION "2024"

[Files]
Source: "{#COVISEDIR}\zebuopt\lib\OpenFOAMInterface.dll"; DestDir: "{app}"; Flags: replacesameversion
Source: "{#COVISEDIR}\src\tools\Revit\OpenFOAM\Families\*"; DestDir: "{app}\Families"; Flags: replacesameversion recursesubdirs
Source: "{#COVISEDIR}\src\tools\Revit\OpenFOAM\BIM\Resources\OpenFOAMInterface.addin"; DestDir: "{userappdata}\Autodesk\Revit\Addins\{#REVIT_VERSION}"; Flags: replacesameversion

[code]
{ HANDLE INSTALL PROCESS STEPS }
procedure CurStepChanged(CurStep: TSetupStep);
var
  AddInFilePath: String;
  LoadedFile : TStrings;
  AddInFileContents: String;
  ReplaceString: String;
  SearchString: String;
begin

  if CurStep = ssPostInstall then
  begin

  AddinFilePath := ExpandConstant('{userappdata}\Autodesk\Revit\Addins\{#REVIT_VERSION}\OpenFOAMInterface.addin');
  LoadedFile := TStringList.Create;
  SearchString := 'Assembly>OpenFOAMInterface.dll<';
  ReplaceString := 'Assembly>' + ExpandConstant('{app}') + '\OpenFOAMInterface.dll<';

  try
      LoadedFile.LoadFromFile(AddInFilePath);
      AddInFileContents := LoadedFile.Text;

      { Only save if text has been changed. }
      if StringChangeEx(AddInFileContents, SearchString, ReplaceString, True) > 0 then
      begin;
        LoadedFile.Text := AddInFileContents;
        LoadedFile.SaveToFile(AddInFilePath);
      end;
    finally
      LoadedFile.Free;
    end;

  end;
end;

[Setup]
AppName=OpenFOAMInterface for Autodesk Revit {#REVIT_VERSION}
AppVerName=OpenFOAMInterface for Autodesk Revit {#REVIT_VERSION}
RestartIfNeededByRun=false
DefaultDirName={commonpf32}\OpenFOAMInterface\{#REVIT_VERSION}
OutputBaseFilename=Setup_OpenFOAMInterface_{#REVIT_VERSION}
ShowLanguageDialog=auto
FlatComponentsList=false
UninstallFilesDir={app}\Uninstall
UninstallDisplayName=OpenFOAMInterface for Autodesk Revit {#REVIT_VERSION}
AppVersion={#REVIT_VERSION}.0
VersionInfoVersion={#REVIT_VERSION}.0
VersionInfoDescription=OpenFOAMInterface for Autodesk Revit {#REVIT_VERSION}
VersionInfoTextVersion=OpenFOAMInterface for Autodesk Revit {#REVIT_VERSION}