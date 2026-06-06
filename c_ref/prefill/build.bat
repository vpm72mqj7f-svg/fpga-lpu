@echo off
REM
REM build.bat — Build cpu_prefill.dll from cpu_prefill.c (Windows)
REM
REM Usage:
REM   build.bat            auto-detect backend
REM   build.bat scalar     portable scalar fallback
REM
REM Requires: MinGW-w64 gcc or MSVC cl.exe
REM Output: build\cpu_prefill.dll
REM

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set BUILD_DIR=%SCRIPT_DIR%build
set SRC=%SCRIPT_DIR%cpu_prefill.c
set OUT=%BUILD_DIR%\cpu_prefill.dll

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM Check for gcc (MinGW)
where gcc >nul 2>&1
if %ERRORLEVEL% equ 0 goto :gcc_build

REM Check for cl (MSVC)
where cl >nul 2>&1
if %ERRORLEVEL% equ 0 goto :msvc_build

echo [ERROR] No C compiler found. Install MinGW-w64 gcc or MSVC.
echo   MinGW: choco install mingw  or  winget install -e --id GnuWin32.Make
exit /b 1

:gcc_build
echo [build.bat] Using gcc (MinGW)
set BACKEND=%1
if "%BACKEND%"=="" set BACKEND=auto

REM Options: auto, scalar (AVX-512 and AMX not available on Windows gcc)
if /I "%BACKEND%"=="scalar" (
    set CFLAGS=-O3 -std=c11 -Wall -Wextra
) else (
    set CFLAGS=-O3 -std=c11 -Wall -Wextra
)

echo [build.bat] Compiling cpu_prefill.c -^> %OUT%
gcc %CFLAGS% -shared -o "%OUT%" "%SRC%" -lpthread -lm

if exist "%OUT%" (
    echo [build.bat] SUCCESS: %OUT% built
    dir "%OUT%"
) else (
    echo [build.bat] ERROR: compilation failed
    exit /b 1
)
goto :eof

:msvc_build
echo [build.bat] Using MSVC cl.exe
set CFLAGS=/O2 /std:c11 /nologo /W3 /LD

echo [build.bat] Compiling cpu_prefill.c -^> %OUT%
cl %CFLAGS% /Fe:"%OUT%" "%SRC%"

if exist "%OUT%" (
    echo [build.bat] SUCCESS: %OUT% built
    dir "%OUT%"
) else (
    echo [build.bat] ERROR: compilation failed
    exit /b 1
)
goto :eof
