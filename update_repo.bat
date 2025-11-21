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
set /p COMMIT_MSG="Enter commit message (or leave empty to skip commit): "
if "!COMMIT_MSG!"=="" (
    echo No commit message provided. Skipping commit.
) else (
    git commit -m "!COMMIT_MSG!"
    if errorlevel 1 (
        echo Commit failed. Possibly no changes to commit.
    )
)

:: Push changes
git push origin main
if errorlevel 1 (
    echo Push failed. Check remote status and authentication.
    exit /b 1
)

echo Repository update completed successfully.
endlocal
