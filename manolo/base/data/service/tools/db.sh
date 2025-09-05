#!/bin/bash
cd "$(dirname "$0")"

# Define variables
MIGRATIONS_FOLDER="../ManoloDataTier.Logic/Migrations"
CONTAINER_NAME="manolo_db"
SEED=false
SEED_SCENARIO=""

# Check for --seed or -s flag
for arg in "$@"; do
  case $arg in
    --seed)
      SEED=true
      shift
      SEED_SCENARIO="$1"  # Capture the scenario after --seed
      shift
      ;;
    -s)
      SEED=true
      shift
      SEED_SCENARIO="$1"  # Capture the scenario after -s
      shift
      ;;
  esac
done

# Check if ida_db container is running
if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
  echo "Error: Container '$CONTAINER_NAME' is already running. Exiting."
  exit 1
fi

# Remove ida_db container if it exists
echo "Removing container '$CONTAINER_NAME' (if it exists)..."
docker rm -f $CONTAINER_NAME 2>/dev/null

# Delete the Migrations folder
echo "Deleting Migrations folder at $MIGRATIONS_FOLDER..."
rm -rf "$MIGRATIONS_FOLDER"

# Add initial migration
echo "Adding initial migration..."
./ef.sh migrations add Initial

# Start Docker Compose with the dev profile
echo "Starting Docker Compose with 'db' profile..."
docker-compose --profile db up -d

# Update the database
echo "Updating the database..."
./ef.sh database update

# Conditionally add the seeding migration
if [ "$SEED" = true ]; then
  echo "Adding seeding migration with scenario: $SEED_SCENARIO..."
  dotnet run --project ../Ida.Data.Seeder -- -c "Host=localhost; Port=9889; Database=ida_db; User Id=postgres; Username=ida; Password=ida" -s "$SEED_SCENARIO"
fi

# Stop Docker Compose
echo "Stopping Docker Compose with 'db' profile..."
docker-compose --profile db stop

echo "Script completed successfully!"
