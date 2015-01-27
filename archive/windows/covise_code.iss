[Code]

program Setup;

var
  CheckInstallForAll, CheckInstallRemoteDaemon, CheckInstallRemoteDaemonForAll: TCheckBox;
  LicenseImportPage: TInputFileWizardPage;
  ConfigImportPage: TInputFileWizardPage;
  CustomPage: TWizardPage;
  Lbl, Lbl2, Lbl3, Lbl4, Lbl5, Lbl6, Lbl11: TLabel;

  advancedInstallation: Boolean;

  Productname, Vendorsupport: String;


#include <./firewall.iss>


function FindCommandlineParam(inParam: String): Boolean;
var
  LoopVar : Integer;
begin
  LoopVar := 0;
  Result := False;
  while ((LoopVar <= ParamCount) and (not Result)) do begin
    if (ParamStr(LoopVar) = inParam) then begin
      Result := True;
      end;
  LoopVar := LoopVar + 1;
  end;
end;





procedure CurStepChanged(CurStep: TSetupStep);
var
   versionstring, filename: String;
begin
if CurStep = ssPostInstall then begin
  // write the version and build number into a file at the end of the installation
  versionstring := '{#VISENSO_VERSION} Build {#BUILDNUMBER}';
  filename := ExpandConstant('{app}\covise\VERSION');
  SaveStringToFile(filename, versionstring, False);
  // add firewall exceptions for COVISE (just the most basic applications)
  AddProgramToFirewall('covise', 'COVISE', ExpandConstant('{app}\covise\amdwin64opt\bin\covise.exe'));
  AddProgramToFirewall('crb', 'COVISE', ExpandConstant('{app}\covise\amdwin64opt\bin\crb.exe'));
  AddProgramToFirewall('mapeditor', 'COVISE', ExpandConstant('{app}\covise\amdwin64opt\bin\mapeditor.exe'));
  AddProgramToFirewall('OpenCOVER', 'COVISE', ExpandConstant('{app}\covise\amdwin64opt\bin\opencover.exe'));
#if (DISTRO_TYPE == "CC")
  // add firewall exceptions for CC
  AddProgramToFirewall('Cyber-Classroom', 'Cyber-Classroom', ExpandConstant('{app}\covise\amdwin64opt\bin\cyberclassroom\bin\cyber-classroom.exe'));
  AddProgramToFirewall('python', 'Cyber-Classroom', ExpandConstant('{app}\covise\extern_libs\Python\DLLs\python.exe')); // probably required for iPad only
  AddPortToFirewall('Cyber-Classroom VR-Controller', 'Cyber-Classroom', 7777, {#NET_FW_IP_PROTOCOL_UDP});
#endif
  end;
end;


procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
if CurUninstallStep = usPostUninstall then begin
  // remove firewall exceptions for COVISE
  RemoveProgramFromFirewall('covise', ExpandConstant('{app}\covise\amdwin64opt\bin\covise.exe'));
  RemoveProgramFromFirewall('crb', ExpandConstant('{app}\covise\amdwin64opt\bin\crb.exe'));
  RemoveProgramFromFirewall('mapeditor', ExpandConstant('{app}\covise\amdwin64opt\bin\mapeditor.exe'));
  RemoveProgramFromFirewall('OpenCOVER', ExpandConstant('{app}\covise\amdwin64opt\bin\opencover.exe'));
#if (DISTRO_TYPE == "CC")
  // remove firewall exceptions for CC
  RemoveProgramFromFirewall('Cyber-Classroom', ExpandConstant('{app}\covise\amdwin64opt\bin\cyberclassroom\bin\cyber-classroom.exe'));
  RemoveProgramFromFirewall('python', ExpandConstant('{app}\covise\extern_libs\Python\DLLs\python.exe'));
  RemovePortFromFirewall('Cyber-Classroom VR-Controller', 7777, {#NET_FW_IP_PROTOCOL_UDP});
#endif
  end;
end;



procedure CurPageChanged(CurPageID: Integer);
begin
if (CurPageID = LicenseImportPage.ID) then begin
  WizardForm.NextButton.Visible := False;
  end;
end;




function InitializeSetup(): Boolean;
begin
#if VERSION == 'VISENSO'
   Vendorsupport := 'support@visenso.de';
#else
  Vendorsupport := 'HLRS';
#endif

Productname := 'COVISE';
#if VERSION == 'VISENSO'
#if (DISTRO_TYPE == "CC")
  Productname := 'Cyber-Classroom';
#elif (DISTRO_TYPE == "KLSM")
  Productname := 'marWORLD 3D';
#endif
#endif

advancedInstallation := FindCommandlineParam('/advanced');

Result := True;
#if (DISTRO_TYPE == "CC")
if not advancedInstallation then begin
  if (FindWindowByWindowName('Cyber-Classroom') <> 0) then Result := False;
  if (FindWindowByWindowName('OpenCOVER') <> 0) then Result := False;
  if (FindWindowByWindowName('Wii HSG-IMIT') <> 0) then Result := False;
  if (FindWindowByWindowName('Tracking') <> 0) then Result := False; // deprecated
  if (FindWindowByWindowName('VR-Controller') <> 0) then Result := False;
  if (not Result) then begin
    MsgBox('At least one process of an already installed Cyber-Classroom is still active.'#13#13'Please end all related processes before installing the new Cyber-Classroom.'#13#13'Contact ' + Vendorsupport + ' if you need any assistance.',  mbInformation, MB_OK );
    exit;
    end;
  end;
#endif
end;




function InstallForAll(): Boolean;
begin

  Result := False;
  if IsAdminLoggedOn then
  begin
     Result := CheckInstallForAll.Checked;
  end;
end;



function InstallForUser(): Boolean;
begin
     Result := NOT InstallForAll();
end;



function InstallRemoteDaemonForAll(): Boolean;
begin
  Result := False;
  if IsAdminLoggedOn then
  begin
     Result := CheckInstallRemoteDaemonForAll.Checked;
  end;
end;




function InstallRemoteDaemonForUser(): Boolean;
begin
     Result := CheckInstallRemoteDaemon.Checked;
end;




function GetWithForwardSlashes(Param: String): String;
var
  mytmpstr : String;
begin
  mytmpstr := ExpandConstant(Param);
  StringChangeEx(mytmpstr, '\', '/', True);
  Result := mytmpstr;
end;




#if VERSION == "VISENSO"
function ShouldSkipPage(PageID: Integer): Boolean;
begin
result := False;

if (advancedInstallation) then exit; // do not skip any pages during advanced installation

// KLS Martin, RTT
#if ( DISTRO_TYPE == "KLSM" || DISTRO_TYPE == "RTT" )
  if (PageID = wpSelectComponents) then
    Result := True;
#endif

// CC
#if (DISTRO_TYPE == "CC")
  if ( (PageID = wpSelectComponents) or (PageID = CustomPage.ID) or (PageID = wpSelectTasks) ) then
    Result := True;
#endif

end;
#endif



#if  (DISTRO_TYPE == "CC")
function NeedRestart(): Boolean;
begin
   Result := True; // we always want a restart after a CC installation
end;
#endif






procedure CreateTheWizardPages;
   var
      appname : String;
      description : String;
begin
  
  appname := 'COVISE';
  description := 'Collaborative Visualisation and Simulation Environment';
#if VERSION == "VISENSO" 
#if DISTRO_TYPE=="CC"
   appname := 'Cyber-Classroom';
   description := 'Immersive Teaching - The New Way of Education';
#elif DISTRO_TYPE == "KLSM"
   appname := 'marWORLD 3D';
   description := 'KLS Martin - marWORLD 3D';
#elif DISTRO_TYPE == "RTT"
   appname := 'COVISE RealFluid';
   description := 'COVISE for RTT RealFluid';
#endif
#endif


  CustomPage := CreateCustomPage(wpWelcome, appname + ' Installation', description);

  Lbl11 := TLabel.Create(CustomPage);
        Lbl11.Caption :='This is a non commercial version of COVISE';
#if GPL_CLEAN == "NO"
#if VERSION == "RRZK"
        Lbl11.Caption :='This is a version of COVISE that is for use by employees of the University of Cologne only. No distribution is allowed.';
#else
        Lbl11.Caption :='This is a non commercial version of COVISE. No distribution is allowed.';
#endif
#else
#if VERSION == "VISENSO"
        Lbl11.Caption :='Commercial version of ' + appname;
#endif
#endif
  Lbl11.AutoSize := True;

  Lbl11.Parent := CustomPage.Surface;

    if IsAdminLoggedOn then
    begin
  CheckInstallForAll := TCheckBox.Create(CustomPage);
  CheckInstallForAll.Top := Lbl11.Top + Lbl11.Height + ScaleY(8);
  CheckInstallForAll.Width := CustomPage.SurfaceWidth;
  CheckInstallForAll.Height := ScaleY(17);
  CheckInstallForAll.Caption := 'Install ' + appname + ' for all users?';
  CheckInstallForAll.Checked := True;
  CheckInstallForAll.Parent := CustomPage.Surface;
    end;

#if (VERSION != "VISENSO") || (DISTRO_TYPE != "CC" && DISTRO_TYPE != "KLSM")
  {note: in Cyber-Classroom installations and for KLS-Martin there is no remote daemon}
  CheckInstallRemoteDaemon := TCheckBox.Create(CustomPage);
    if IsAdminLoggedOn then
    begin
  CheckInstallRemoteDaemon.Top := CheckInstallForAll.Top + CheckInstallForAll.Height + ScaleY(8);
  end
  else begin
  CheckInstallRemoteDaemon.Top := Lbl11.Top + Lbl11.Height + ScaleY(8);
  end;
  CheckInstallRemoteDaemon.Width := CustomPage.SurfaceWidth;
  CheckInstallRemoteDaemon.Height := ScaleY(17);
  CheckInstallRemoteDaemon.Checked := False;
  CheckInstallRemoteDaemon.Caption := 'Start COVISE Daemon automatically for current user?';
  CheckInstallRemoteDaemon.Parent := CustomPage.Surface;

    if IsAdminLoggedOn then
    begin
  CheckInstallRemoteDaemonForAll := TCheckBox.Create(CustomPage);
  CheckInstallRemoteDaemonForAll.Top := CheckInstallRemoteDaemon.Top + CheckInstallRemoteDaemon.Height + ScaleY(8);
  CheckInstallRemoteDaemonForAll.Width := CustomPage.SurfaceWidth;
  CheckInstallRemoteDaemonForAll.Height := ScaleY(17);
  CheckInstallRemoteDaemonForAll.Checked := False;
  CheckInstallRemoteDaemonForAll.Caption := 'Start COVISE Daemon automatically for all users?';
  CheckInstallRemoteDaemonForAll.Parent := CustomPage.Surface;

    end;
#endif


end;






procedure AboutButtonOnClick(Sender: TObject);
begin
#if VERSION != "VISENSO"
  MsgBox('This Wizard allows to install and configure COVISE', mbInformation, mb_Ok);
#else
#if (DISTRO_TYPE == "CC")
  MsgBox('Cyber-Classroom'#13#13'{#SUFFIX_ARCH} Version'#13#13 + 
     'mail to info@visenso.de'#13#13'(C) 2009 VISENSO GmbH', mbInformation, mb_Ok);
#elif (DISTRO_TYPE == "RTT")
  MsgBox('COVISE'#13'(Collaborative Visualisation and Simulation Environment)'#13'for RTT RealFluid'#13#13'{#SUFFIX_ARCH} Version'#13#13 + 
     'mail to info@visenso.de'#13#13'(C) 2003 VISENSO GmbH', mbInformation, mb_Ok);
#elif (DISTRO_TYPE == "KLSM")
  MsgBox('COVISE'#13'(Collaborative Visualisation and Simulation Environment)'#13'for KLS Martin'#13#13'{#SUFFIX_ARCH} Version'#13#13 + 
     'mail to info@visenso.de'#13#13'(C) 2003 VISENSO GmbH', mbInformation, mb_Ok);
#else
  MsgBox('COVISE'#13'(Collaborative Visualisation and Simulation Environment)'#13#13'{#SUFFIX_ARCH} Version'#13#13 + 
     'mail to info@visenso.de'#13#13'(C) 2003 VISENSO GmbH', mbInformation, mb_Ok);
#endif
#endif
end;



procedure URLLabelOnClick(Sender: TObject);
var
  ErrorCode: Integer;
begin
#if VERSION == "VISENSO"
#if  (DISTRO_TYPE == "CC")
  ShellExec('open', 'http://www.cyber-classroom.de/', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
#else
  ShellExec('open', 'http://www.visenso.de', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
#endif
#else
  ShellExec('open', 'http://www.hlrs.de/organization/vis', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
#endif
end;







/////////////////////////////////// BEGIN License copying ////////////////////////////////////////////





procedure PerformCopyLicense(Sender: TObject);
var
   LicenseFile: String;
   TrgLicenseFile: String;
   CCFile: String;
   TrgCCFile: String;     
   TrackingScript: String;
   TrgTrackingScript: String;        
   CopySuccess: Boolean;
   nRun: Integer;   
begin
  
   { load the license to be imported}
   LicenseFile := LicenseImportPage.Values[0];
   if ((VarIsNull(LicenseFile)) or (not FileExists(LicenseFile))) then begin
      if VarIsNull(LicenseFile) then
         LicenseFile := '<null>';
      MsgBox('File ' + LicenseFile + ' does not seem to exist.'#13#13'Please select a valid license file.'#13#13'Contact ' + Vendorsupport + ' if you need any assistance.', mbError, mb_Ok);
      Exit;
   end;
   LicenseFile := ExpandConstant(LicenseFile);
   
   { try to copy license file }
   TrgLicenseFile := ExpandConstant('{app}\covise\config\config.license.xml');
   CopySuccess := FileCopy(LicenseFile, TrgLicenseFile, True);
   if (CopySuccess = False) then begin
      // note: if CopySuccess = False then it is normally due to config.license.xml already existing. 
      //    This happens whenever COVISE gets installed over an existing installation.
      // make a save copy and retry
      nRun := 0;
      while ((nRun < 100) and (not CopySuccess)) do begin
         CopySuccess := FileCopy(ExpandConstant(TrgLicenseFile), ExpandConstant('{app}\covise\config\config.license.' + IntToStr(nRun) + '.bak'), True);
	 nRun := nRun + 1;
      end;
      if (CopySuccess = True) then begin // could copy existing license -> go on and overwrite
	     //MsgBox('made a backup of '+TrgLicenseFile, mbInformation, mb_Ok);
         CopySuccess := FileCopy(LicenseFile, ExpandConstant(TrgLicenseFile), False);
      end;
   end;

   if (CopySuccess = True) then begin
      MsgBox('Successfully activated ' + Productname + ' license!', mbInformation, mb_Ok);
   end else begin
      MsgBox('Error: COVISE license activation failed!'#13'Please clean up config folder. You have got too much license copies.'#13#13'Please contact ' + 
             Vendorsupport + ' to solve this problem.', mbInformation, mb_Ok);
      Exit; 
   end;

WizardForm.NextButton.Visible := True;
WizardForm.NextButton.OnClick(nil);
end;   




procedure PerformSkipLicense(Sender: TObject);
begin
if MsgBox('You need a valid license in order to run ' + Productname + '. Do you really want to skip this step?', mbConfirmation, MB_YESNO) = IDYES then begin
  WizardForm.NextButton.Visible := True;
  WizardForm.NextButton.OnClick(nil);
  end;
end;





 
{if user has a valid license, let her import it at this page}
procedure ImportCOVISElicense();
var
   ImportButton, SkipButton : TButton;
begin
   // place a custom wizard page after the installing page, so that user can
   // select and import license
   LicenseImportPage := CreateInputFilePage(wpInstalling,
     'Select license file to be imported.', 'Here you can import a valid ' + Productname + ' license.',
     'Please select the ' + Productname + ' license file, then click the import button.');
   LicenseImportPage.Add('Location of license file:',         // caption
      'XML files|*.xml|All files|*.*',    // filters
      '.xml');// default extension
   LicenseImportPage.Values[0] := ExpandConstant('{src}\config.license.xml');
   
   ImportButton := TButton.Create(LicenseImportPage);
   ImportButton.Width := 120;
   ImportButton.Height := 28;
   ImportButton.Left := LicenseImportPage.SurfaceWidth - ImportButton.Width;
   ImportButton.Top := 200;
   ImportButton.Caption := '&Import License';
   ImportButton.OnClick := @PerformCopyLicense;
   ImportButton.Parent := LicenseImportPage.Surface;

   SkipButton := TButton.Create(LicenseImportPage);
   SkipButton.Width := 80;
   SkipButton.Height := 28;
   SkipButton.Left := ImportButton.Left - 12 - SkipButton.Width;
   SkipButton.Top := 200;
   SkipButton.Caption := '&Skip';
   SkipButton.OnClick := @PerformSkipLicense;
   SkipButton.Parent := LicenseImportPage.Surface;
end;













/////////////////////////////////// END License copying ////////////////////////////////////////////






procedure InitializeWizard();
var
  AboutButton, CancelButton: TButton;
  URLLabel: TNewStaticText;
  BackgroundBitmapImage: TBitmapImage;
  BackgroundBitmapText: TNewStaticText;
  i: Integer;
begin


  { Custom wizard pages }

  CreateTheWizardPages;

  { Other custom controls }

  CancelButton := WizardForm.CancelButton;

  AboutButton := TButton.Create(WizardForm);
  AboutButton.Left := WizardForm.ClientWidth - CancelButton.Left - CancelButton.Width;
  AboutButton.Top := CancelButton.Top;
  AboutButton.Width := CancelButton.Width;
  AboutButton.Height := CancelButton.Height;
  AboutButton.Caption := '&About...';
  AboutButton.OnClick := @AboutButtonOnClick;
  AboutButton.Parent := WizardForm;

  URLLabel := TNewStaticText.Create(WizardForm);
#if VERSION == "VISENSO"
#if  (DISTRO_TYPE == "CC")
  URLLabel.Caption := 'visit Cyber-Classroom.de';
#else
  URLLabel.Caption := 'visit VISENSO.de';
#endif
#else
  URLLabel.Caption := 'www.hlrs.de';
#endif
  URLLabel.Cursor := crHand;
  URLLabel.OnClick := @URLLabelOnClick;
  URLLabel.Parent := WizardForm;
  { Alter Font *after* setting Parent so the correct defaults are inherited first }
  URLLabel.Font.Style := URLLabel.Font.Style + [fsUnderline];
  URLLabel.Font.Color := clBlue;
  URLLabel.Top := AboutButton.Top + AboutButton.Height - URLLabel.Height - 2;
  URLLabel.Left := AboutButton.Left + AboutButton.Width + ScaleX(20);

  BackgroundBitmapImage := TBitmapImage.Create(MainForm);
  BackgroundBitmapImage.Left := 10;
  BackgroundBitmapImage.Top := 100;
  BackgroundBitmapImage.AutoSize := True;
  BackgroundBitmapImage.Bitmap := WizardForm.WizardBitmapImage.Bitmap;
  BackgroundBitmapImage.Parent := MainForm;

  BackgroundBitmapText := TNewStaticText.Create(MainForm);
  BackgroundBitmapText.Left := BackgroundBitmapImage.Left;
  BackgroundBitmapText.Top := BackgroundBitmapImage.Top + BackgroundBitmapImage.Height + ScaleY(8);
  BackgroundBitmapText.Caption := 'TBitmapImage';
  BackgroundBitmapText.Parent := MainForm;
  
#if VERSION == "VISENSO"
#if (DISTRO_TYPE == "PLAINVANILLA" || DISTRO_TYPE == "CC")
  ImportCOVISElicense();
#endif
#endif

// pre-select all tasks in a cyber-classroom installation
#if (DISTRO_TYPE == "CC")
  for i := 0 to WizardForm.TasksList.Items.Count-1 do WizardForm.TasksList.Checked[i] := True;
#endif

end;





begin
   {body of program Setup}
end.
