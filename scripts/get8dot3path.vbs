Set fso=CreateObject("Scripting.FileSystemObject")

' is object a file or folder?
If fso.FolderExists(WScript.Arguments(0)) Then
   ' it´s a folder
   Set objFolder = fso.GetFolder(WScript.Arguments(0))
   WScript.echo(objFolder.ShortPath)
End If

If fso.FileExists(WScript.Arguments(0)) Then
   ' it´s a file
   Set objFile = fso.GetFile(WScript.Arguments(0))
   WScript.echo(objFile.ShortPath)
End If