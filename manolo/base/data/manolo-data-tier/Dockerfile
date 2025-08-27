# --------------------------------------------------------------------------------
# 1. Build Stage
# --------------------------------------------------------------------------------
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy solution and project files only (cache restore layer)
COPY *.sln ./
COPY ManoloDataTier.Api/*.csproj ./ManoloDataTier.Api/
COPY ManoloDataTier.Common/*.csproj ./ManoloDataTier.Common/
COPY ManoloDataTier.Logic/*.csproj ./ManoloDataTier.Logic/
COPY ManoloDataTier.Storage/*.csproj ./ManoloDataTier.Storage/

# Restore dependencies (cached)
RUN --mount=type=cache,target=/root/.nuget/packages \
    dotnet restore

# Copy only source files for the API project
COPY ManoloDataTier.Api ./ManoloDataTier.Api
COPY ManoloDataTier.Common ./ManoloDataTier.Common
COPY ManoloDataTier.Logic ./ManoloDataTier.Logic
COPY ManoloDataTier.Storage ./ManoloDataTier.Storage

# Build & publish in one step (cached NuGet)
RUN --mount=type=cache,target=/root/.nuget/packages \
    dotnet publish ./ManoloDataTier.Api/ManoloDataTier.Api.csproj \
        -c Release \
        -o /app/publish \
        --no-restore

# --------------------------------------------------------------------------------
# 2. Runtime Stage
# --------------------------------------------------------------------------------
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app

COPY --from=build /app/publish ./

EXPOSE 5001
EXPOSE 5002 

ENTRYPOINT ["dotnet", "ManoloDataTier.Api.dll"]
