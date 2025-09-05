using Microsoft.AspNetCore.Builder;
using Serilog;

namespace ManoloDataTier.Logic.Extensions;

public static class LoggingExtensions{

    public static void UseLogging(this WebApplicationBuilder builder){
        builder.Host.UseSerilog((_, loggerConfiguration) => {
            loggerConfiguration.ReadFrom.Configuration(builder.Configuration)
                               .Enrich.WithCorrelationIdHeader(headerKey: "x-cid");
        });
    }

}