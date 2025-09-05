@echo off
setlocal

:: Define global variables for the paths
set STORAGE_PROJECT=..\ManoloDataTier.Logic
set API_PROJECT=..\ManoloDataTier.Api
set CONTAINER_NAME=manolo_db
set MIGRATIONS_FOLDER=%STORAGE_PROJECT%\Migrations

:: Navigate to the script directory
cd /d "%~dp0" || exit /b 1

:: Path to the dotnet-ef tool
set EF_TOOL=%USERPROFILE%\.dotnet\tools\dotnet-ef.exe

:: Ensure the dotnet-ef tool is available
if not exist "%EF_TOOL%" (
  echo Error: dotnet-ef tool not found at %EF_TOOL%
  exit /b 1
)

:: Check if manolo_db container is running
for /f "delims=" %%i in ('docker ps --filter "name=%CONTAINER_NAME%" --format "{{.Names}}"') do (
  set CONTAINER_RUNNING=%%i
)

if defined CONTAINER_RUNNING (
  echo Error: Container "%CONTAINER_NAME%" is already running. Exiting.
  exit /b 1
)

:: Remove manolo_db container if it exists
echo Removing container "%CONTAINER_NAME%" (if it exists)...
docker rm -f %CONTAINER_NAME% 2>nul

:: Delete the Migrations folder
echo Deleting Migrations folder at %MIGRATIONS_FOLDER%...
rmdir /s /q "%MIGRATIONS_FOLDER%"

:: Recreate the migration
echo Creating a new migration...
"%EF_TOOL%" migrations add InitialMigration --project %STORAGE_PROJECT% --startup-project %API_PROJECT%

:: Start Docker Compose with the dev profile
echo Starting Docker Compose with "db" profile...
docker-compose --profile dev up -d

:: Apply the migration to update the database
echo Applying the new migration to the database...
"%EF_TOOL%" database update --project %STORAGE_PROJECT% --startup-project %API_PROJECT%

:: Stop Docker Compose
echo Stopping Docker Compose with "db" profile...
docker-compose --profile dev stop

echo Migration and database update completed successfully!

endlocal
