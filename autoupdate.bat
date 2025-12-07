@echo off
cd /d "%~dp0"
echo ==========================================
echo      Quanta Futures Auto-Update & Repair
echo ==========================================
echo.

REM Check if git is available
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo Git not found in PATH. Trying standard locations...
    if exist "C:\Program Files\Git\cmd\git.exe" (
        set "GIT_CMD=C:\Program Files\Git\cmd\git.exe"
    ) else (
        echo ERROR: Git is not installed or not found.
        pause
        exit /b
    )
) else (
    set "GIT_CMD=git"
)

echo Using Git: %GIT_CMD%
echo.

echo 1. Fetching latest data from GitHub...
"%GIT_CMD%" fetch origin
if %errorlevel% neq 0 (
    echo Error fetching from origin. Check your internet connection.
    pause
    exit /b
)

echo.
echo 2. Resetting local changes to match remote (Repairing)...
REM This discards local changes to tracked files to ensure a clean update
"%GIT_CMD%" reset --hard origin/main

echo.
echo 3. Pulling latest updates...
"%GIT_CMD%" pull origin main

echo.
echo ==========================================
echo           Update Complete!
echo ==========================================
pause
