# --------------------------------------------------------------------------------
# 1. Build Stage
# --------------------------------------------------------------------------------
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

ARG USE_CACHE=true

COPY *.sln ./
COPY ManoloDataTier.Api/*.csproj ./ManoloDataTier.Api/
COPY ManoloDataTier.Common/*.csproj ./ManoloDataTier.Common/
COPY ManoloDataTier.Logic/*.csproj ./ManoloDataTier.Logic/
COPY ManoloDataTier.Storage/*.csproj ./ManoloDataTier.Storage/

# Restore dependencies
RUN if [ "$USE_CACHE" = "true" ]; then \
      dotnet restore --packages /root/.nuget/packages ; \
    else \
      dotnet restore ; \
    fi

COPY ManoloDataTier.Api ./ManoloDataTier.Api
COPY ManoloDataTier.Common ./ManoloDataTier.Common
COPY ManoloDataTier.Logic ./ManoloDataTier.Logic
COPY ManoloDataTier.Storage ./ManoloDataTier.Storage

# Build & publish
RUN if [ "$USE_CACHE" = "true" ]; then \
      dotnet publish ./ManoloDataTier.Api/ManoloDataTier.Api.csproj \
        -c Release \
        -o /app/publish \
        --no-restore --packages /root/.nuget/packages ; \
    else \
      dotnet publish ./ManoloDataTier.Api/ManoloDataTier.Api.csproj \
        -c Release \
        -o /app/publish \
        --no-restore ; \
    fi

# --------------------------------------------------------------------------------
# 2. Runtime Stage
# --------------------------------------------------------------------------------
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app

COPY --from=build /app/publish ./

EXPOSE 5001
EXPOSE 5002

ENTRYPOINT ["dotnet", "ManoloDataTier.Api.dll"]

