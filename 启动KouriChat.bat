@echo off
chcp 65001 >nul
title KouriChat 一键管理员启动脚本

:: 检查并请求管理员权限
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo [INFO] 正在自动请求管理员提权...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    cd /d "%~dp0"
    echo ==================================================
    echo         KouriChat - AI 微信机器人 Web 控制台
    echo ==================================================
    echo [SUCCESS] 已成功获得管理员权限！
    echo [INFO] 正在启动 run_config_web.py ...
    echo.
    .venv\Scripts\python.exe run_config_web.py
    pause
