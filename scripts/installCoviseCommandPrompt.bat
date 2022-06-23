:: edit your filepaths and make sure the linkpath exists:

set linkpath=%HOMEPATH%\Desktop
set linkname=test
set pathToCmd=C:\Windows\System32
set ARCHSUFFIX=zebuopt
set COVISEDIR=C:\src\covise
:: users should not edit what follows 
set arguments=/k %COVISEDIR%\Scripts\coviseenv.bat %COVISEDIR% %ARCHSUFFIX% 
set cmdExe=cmd
set workingDir=%HOMEDRIVE%%HOMEPATH%
if not exist "%linkpath%" md "%linkpath%"
:: create temporary VBScript ...
echo archsuffix="%ARCHSUFFIX%">>%temp%\MakeShortCut.vbs
echo found=instr(len(archsuffix) - 3, archsuffix, "opt", 0)>>%temp%\MakeShortCut.vbs
echo if found then archsuffix=left(archsuffix, found - 1)>>%temp%\MakeShortCut.vbs
echo Set objShell=WScript.CreateObject("Wscript.Shell")>>%temp%\MakeShortCut.vbs
echo Set objShortcut=objShell.CreateShortcut("%linkpath%\%linkname%.lnk")>>%temp%\MakeShortCut.vbs
echo objShortcut.TargetPath="%pathToCmd%\%cmdExe%.exe">>%temp%\MakeShortCut.vbs
echo objShortcut.Arguments="%arguments%" + " %COVISEDIR%\..\externlibs\" + archsuffix>>%temp%\MakeShortCut.vbs
echo objShortcut.Description="%description%">>%temp%\MakeShortCut.vbs
echo objShortcut.WorkingDirectory="%workingDir%">>%temp%\MakeShortCut.vbs
echo objShortcut.Save>>%temp%\MakeShortCut.vbs

::... run it ...
cscript //nologo %temp%\MakeShortCut.vbs

::... and delete it.
del %temp%\MakeShortCut.vbs