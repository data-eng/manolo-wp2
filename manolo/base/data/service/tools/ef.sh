#!/usr/bin/env bash
cd "$(dirname "$0")"
~/.dotnet/tools/dotnet-ef "$@" --project ../ManoloDataTier.Logic --startup-project ../ManoloDataTier.Api