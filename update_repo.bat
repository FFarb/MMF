@echo off
setlocal enabledelayedexpansion

:: Change to project directory
cd /d "c:\Users\chern\Desktop\projects\quanta futures"

:: Ensure we are on the main branch (adjust if different)
git checkout main

:: Pull latest changes with rebase to avoid merge commits
git pull --rebase origin main
if errorlevel 1 (
    echo Pull failed. Resolve conflicts manually before running the script again.
    exit /b 1
)

:: Stage all changes
git add .

:: Prompt for commit message
set /p COMMIT_MSG="Enter commit message (or press Enter for default): "
if "!COMMIT_MSG!"=="" (
    :: Generate default commit message with timestamp
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set DATE=%%a-%%b-%%c
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set TIME=%%a:%%b
    set COMMIT_MSG=Auto-update: !DATE! !TIME!
    echo Using default commit message: !COMMIT_MSG!
)
git commit -m "!COMMIT_MSG!"
if errorlevel 1 (
    echo Commit failed. Possibly no changes to commit.
)

:: Push changes
git push origin main
if errorlevel 1 (
    echo Push failed. Check remote status and authentication.
    exit /b 1
)

echo Repository update completed successfully.
endlocal
