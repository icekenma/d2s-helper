@echo off
rem D2S Helper: installs the Korean OCR of Windows (reads item info boxes: skill lines, names)
rem Needs administrator rights once (a Windows prompt asks for it).
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Asking for administrator rights...
  powershell -NoProfile -Command "Start-Process -FilePath \"%~f0\" -Verb RunAs"
  exit /b
)
echo Installing Korean OCR (Language.OCR ko-KR). This can take a few minutes...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-WindowsCapability -Online | Where-Object { $_.Name -like \"Language.OCR*ko-KR*\" } | Add-WindowsCapability -Online | Out-Null"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-WindowsCapability -Online | Where-Object { $_.Name -like \"Language.OCR*ko-KR*\" } | Format-Table Name,State -AutoSize"
echo.
echo If State is Installed: done. D2S Helper uses it within 5 minutes (no restart needed).
pause
