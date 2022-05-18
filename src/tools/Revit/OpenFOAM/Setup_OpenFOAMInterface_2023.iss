#define COVISEDIR GetEnv("COVISEDIR")

[Files]
Source: "{#COVISEDIR}\zebuopt\lib\OpenFOAMInterface.dll"; DestDir: "{app}"; Flags: replacesameversion
Source: "{#COVISEDIR}\src\tools\Revit\OpenFOAM\Families\*"; DestDir: "{app}\Families"; Flags: replacesameversion recursesubdirs
Source: "{#COVISEDIR}\src\tools\Revit\OpenFOAM\BIM\Resources\OpenFOAMInterface.addin"; DestDir: "{userappdata}\Autodesk\Revit\Addins\2023"; Flags: replacesameversion

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

  AddinFilePath := ExpandConstant('{userappdata}\Autodesk\Revit\Addins\2023\OpenFOAMInterface.addin');
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
AppName=OpenFOAMInterface for Autodesk Revit 2023
AppVerName=OpenFOAMInterface for Autodesk Revit 2023
RestartIfNeededByRun=false
DefaultDirName={commonpf32}\OpenFOAMInterface\2023
OutputBaseFilename=Setup_OpenFOAMInterface_2023
ShowLanguageDialog=auto
FlatComponentsList=false
UninstallFilesDir={app}\Uninstall
UninstallDisplayName=OpenFOAMInterface for Autodesk Revit 2023
AppVersion=2023.0
VersionInfoVersion=2023.0
VersionInfoDescription=OpenFOAMInterface for Autodesk Revit 2023
VersionInfoTextVersion=OpenFOAMInterface for Autodesk Revit 2023