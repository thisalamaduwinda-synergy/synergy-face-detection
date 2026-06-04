Set fso = CreateObject("Scripting.FileSystemObject")
appDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = appDir & "\.venv\Scripts\pythonw.exe"

Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = appDir

If fso.FileExists(pythonExe) Then
  shell.Run Chr(34) & pythonExe & Chr(34) & " " & Chr(34) & appDir & "\main_qt.py" & Chr(34), 0, False
Else
  shell.Run "pythonw " & Chr(34) & appDir & "\main_qt.py" & Chr(34), 0, False
End If
