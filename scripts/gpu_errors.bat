@echo off
rem The last GPU / power events from the Windows System log, local date and time, newest first.
rem   scripts\gpu_errors.bat        last 10
rem   scripts\gpu_errors.bat N      last N, e.g. scripts\gpu_errors.bat 3
rem Read only, no administrator rights needed. GPUID 2e00 = the card on PCI bus 2E (nvidia-smi --query-gpu=pci.bus_id).

setlocal
set N=%~1
if "%N%"=="" set N=10
echo %N%| findstr /r "^[1-9][0-9]*$" >nul || (echo N must be a whole number, e.g. scripts\gpu_errors.bat 5 & exit /b 1)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm','Display','Microsoft-Windows-WHEA-Logger','Microsoft-Windows-Kernel-Power'} -ErrorAction SilentlyContinue |" ^
  "  Where-Object { $_.ProviderName -in 'nvlddmkm','Microsoft-Windows-WHEA-Logger' -or ($_.ProviderName -eq 'Display' -and $_.Id -eq 4101) -or ($_.ProviderName -eq 'Microsoft-Windows-Kernel-Power' -and $_.Id -eq 41) } |" ^
  "  Select-Object -First %N% | ForEach-Object {" ^
  "    $p = @($_.Properties | ForEach-Object { $_.Value });" ^
  "    $what = switch ($_.ProviderName) { 'nvlddmkm' { $p[1] } 'Microsoft-Windows-Kernel-Power' { 'PC lost power / restarted without shutting down' } default { ($_.Message -split \"`n\")[0].Trim() } };" ^
  "    '{0}   {1}' -f $_.TimeCreated.ToString('ddd dd-MM-yyyy HH:mm:ss'), $what }"

echo %cmdcmdline% | find /i "%~0" >nul && pause
endlocal
