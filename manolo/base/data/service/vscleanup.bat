@echo off
setlocal

:: Will delete all .vs , bin and obj folder under the current directory
set foldersToDelete=".vs" "bin" "obj"


for %%F in (%foldersToDelete%) do (
    echo Deleting any folder named %%F...

    :: We use 'for /d /r' to find directories that matching the current name and delete them
    for /r %%D in (.) do (
        if exist "%%D\%%F" (
            echo Deleting folder: %%D\%%F
            rmdir /s /q "%%D\%%F"
        )
    )
)

echo Deletion complete.
endlocal
pause
