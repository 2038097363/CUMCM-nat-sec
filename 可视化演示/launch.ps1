$ErrorActionPreference = 'Stop'
try {
    Set-Location -LiteralPath $PSScriptRoot
    try {
        $existing = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/status' -TimeoutSec 2
        if ($null -ne $existing.state -and $null -ne $existing.events) {
            Start-Process 'http://127.0.0.1:8765/'
            return
        }
    } catch { }
    $python = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $knownPython = Join-Path $env:USERPROFILE 'anaconda3\python.exe'
    if ((Test-Path -LiteralPath $knownPython) -and (-not $python -or $python.Source -like '*WindowsApps*')) {
        $pythonPath = $knownPython
    } elseif ($python) {
        $pythonPath = $python.Source
    } else {
        throw '没有找到 Python，请安装依赖后重试。'
    }
    Write-Host '正在启动本地演示，浏览器将自动打开。关闭本窗口即可停止服务。'
    & $pythonPath -X utf8 (Join-Path $PSScriptRoot 'server.py')
    if ($LASTEXITCODE -ne 0) { throw "演示服务退出，代码 $LASTEXITCODE。" }
} catch {
    Write-Host "启动失败：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host '如果提示端口占用，先关闭之前的演示窗口，再重新启动。'
    Read-Host '按回车关闭窗口'
}
