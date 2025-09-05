using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Configuration;

namespace ManoloDataTier.Logic.Extensions;

public static class DatabaseConfiguration{

    public static void AddConfiguration(this WebApplicationBuilder builder){
        var env            = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? builder.Environment.EnvironmentName;
        var configRootPath = Environment.GetEnvironmentVariable("CONFIG_ROOT") ?? "Configs";
        var extraConfig    = Environment.GetEnvironmentVariable("EXTRA_CONFIG_NAME");

        builder.Configuration.AddEnvironmentVariables();
        builder.Configuration.AddJsonFile($"{configRootPath}/appsettings.{env}.json", false, true);
        builder.Configuration.AddJsonFile($"{configRootPath}/logging.{env}.json", true, true);

        if (extraConfig != null)
            builder.Configuration.AddJsonFile($"{configRootPath}/appsettings.{extraConfig}.json", false, true);
    }

}