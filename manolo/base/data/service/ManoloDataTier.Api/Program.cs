using System.Reflection;
using FluentValidation;
using FluentValidation.AspNetCore;
using ManoloDataTier.Api.Controllers;
using ManoloDataTier.Api.SignalRFeatures.Item.GetItemData;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Extensions;
using ManoloDataTier.Logic.Interfaces;
using ManoloDataTier.Logic.Services;
using ManoloDataTier.Logic.Settings;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Mvc.Controllers;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.EntityFrameworkCore;
using Microsoft.Identity.Web;
using Npgsql;

var builder = WebApplication.CreateBuilder(args);

//Configuration & logging
builder.AddConfiguration();
builder.UseLogging();

//Add SignalR
builder.Services.AddSignalR();
builder.Services.AddScoped<GetItemDataSignalR>();

var env = builder.Environment;

string connectionString;
string mlflowUrl;

if (env.IsProduction())
{
    string dbPassword;

    var dbPasswordFile = Environment.GetEnvironmentVariable("DB_PASSWORD_FILE");

    if (!string.IsNullOrEmpty(dbPasswordFile) && File.Exists(dbPasswordFile))
        dbPassword = File.ReadAllText(dbPasswordFile)
                         .Trim();
    else
        dbPassword = Environment.GetEnvironmentVariable("DB_PASSWORD") ?? "manolo";

    var dbHost = Environment.GetEnvironmentVariable("DB_HOST") ?? "localhost";
    var dbPort = Environment.GetEnvironmentVariable("DB_PORT") ?? "8432";
    var dbName = Environment.GetEnvironmentVariable("DB_NAME") ?? "manolo_db";
    var dbUser = Environment.GetEnvironmentVariable("DB_USERNAME") ?? "manolo";
    mlflowUrl = Environment.GetEnvironmentVariable("MLFLOW_URL") ?? string.Empty;

    connectionString =
        $"Host={dbHost};Port={dbPort};Database={dbName};Username={dbUser};Password={dbPassword}";
}
else
{
    connectionString = builder.Configuration.GetConnectionString("DefaultConnection")!;

    mlflowUrl = builder.Configuration
                       .GetSection("ConnectionStrings")["MlflowUrl"]
             ?? string.Empty;
}

builder.Configuration["ConnectionStrings:DefaultConnection"] = connectionString;
builder.Configuration["ConnectionStrings:MlflowUrl"]         = mlflowUrl;

var host = Environment.GetEnvironmentVariable("HTTP3_HOST") ?? "localhost";

var port = int.TryParse(Environment.GetEnvironmentVariable("HTTP3_PORT"), out var p)
    ? p
    : 5001;

builder.WebHost.ConfigureKestrel
((context, options) =>
    {
        options.Limits.MaxRequestBodySize    = 21474836480; // 20 GB (in bytes)
        options.Limits.RequestHeadersTimeout = TimeSpan.FromMinutes(10);

        var certPath = Environment.GetEnvironmentVariable("CERTIFICATE_PATH")
                    ?? context.Configuration["Kestrel:Certificates:Default:Path"];

        var certPassword = Environment.GetEnvironmentVariable("CERTIFICATE_PASSWORD")
                        ?? context.Configuration["Kestrel:Certificates:Default:Password"];

        if (!string.IsNullOrEmpty(certPath) && !string.IsNullOrEmpty(certPassword))
        {
            options.ListenAnyIP
            (
                port,
                listenOptions =>
                {
                    listenOptions.Protocols = HttpProtocols.Http1AndHttp2AndHttp3;
                    listenOptions.UseHttps(certPath, certPassword);
                    listenOptions.UseConnectionLogging();
                }
            );
        }
        else
        {
            options.ListenAnyIP
            (
                port,
                listenOptions =>
                {
                    listenOptions.Protocols = HttpProtocols.Http1AndHttp2;
                    listenOptions.UseConnectionLogging();
                }
            );
        }
    }
);

builder.Services.Configure<FormOptions>
(options =>
    {
        options.MultipartBodyLengthLimit = 21474836480; // 20 GB (in bytes)
    }
);

// Add services to the container.
builder.Services
       .AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
       .AddMicrosoftIdentityWebApi(builder.Configuration.GetSection("AzureAd"));

//Add Controllers
#pragma warning disable CS0618 // Type or member is obsolete
builder.Services
       .AddControllers()
       .AddFluentValidation
       (config =>
           {
               config.RegisterValidatorsFromAssembly(Assembly.GetExecutingAssembly());
               config.ImplicitlyValidateChildProperties        = true;
               config.ImplicitlyValidateRootCollectionElements = true;
           }
       );
#pragma warning restore CS0618 // Type or member is obsolete

//Add Swagger Config
builder.Services.AddSwaggerGen
(c =>
    {
        c.TagActionsBy
        (api =>
            {
                if (api.GroupName != null)
                    return [api.GroupName,];

                if (api.ActionDescriptor is ControllerActionDescriptor controllerActionDescriptor)
                    return [controllerActionDescriptor.ControllerName,];

                throw new InvalidOperationException("Unable to determine tag for endpoint.");
            }
        );

        c.DocInclusionPredicate((_, _) => true);
    }
);

//Add MediatR and Validation
builder.Services.AddValidatorsFromAssembly(Assembly.GetExecutingAssembly());
builder.Services.AddMediatR(cfg => { cfg.RegisterServicesFromAssembly(Assembly.GetExecutingAssembly()); });

// Add DbContext
builder.Services.AddDbContext<ManoloDbContext>
(options =>
     options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection"))
);

// Mlflow
builder.Services.Configure<MlflowSettings>(builder.Configuration.GetSection("ConnectionStrings"));
builder.Services.AddHttpClient<MlflowService>();
builder.Services.AddScoped<MlflowService>();

// Add Services
builder.Services.AddScoped<ServiceHelpers>();

#pragma warning disable CS0618 // Type or member is obsolete
NpgsqlConnection.GlobalTypeMapper
#pragma warning restore CS0618 // Type or member is obsolete
                .EnableDynamicJson();

// Register the DynamicTableService
builder.Services.AddScoped<DynamicTableService>();

//Seed
builder.Services.AddScoped<UserSeeder>();

// Register the IdResolverService
builder.Services.AddScoped<IIdResolverService, IdResolverService>();

builder.Services.AddEndpointsApiExplorer();

// Add authentication and cookie configuration
builder.Services
       .AddAuthentication("Cookies")
       .AddCookie
       (
           "Cookies",
           options =>
           {
               options.LoginPath         = "/login";
               options.LogoutPath        = "/logout";
               options.AccessDeniedPath  = "/denied";
               options.ExpireTimeSpan    = TimeSpan.FromMinutes(60);
               options.SlidingExpiration = true;

               options.Events.OnRedirectToAccessDenied = context =>
               {
                   context.Response.StatusCode  = StatusCodes.Status403Forbidden;
                   context.Response.ContentType = "application/json";

                   var errorResponse = Result.Failure(DomainError.UserAccessLevelNotAuthorized);

                   return context.Response.WriteAsJsonAsync(errorResponse.Message);
               };

               options.Events.OnRedirectToLogin = context =>
               {
                   context.Response.StatusCode  = StatusCodes.Status401Unauthorized;
                   context.Response.ContentType = "application/json";

                   var errorResponse = Result.Failure(DomainError.UserNotLoggedIn);

                   return context.Response.WriteAsJsonAsync(errorResponse.Message);
               };
           }
       );

builder.Services
       .AddAuthorizationBuilder()
       .AddPolicy
       (
           "AdminOnly",
           policy =>
               policy.RequireAssertion
               (context =>
                   {
                       var accessLevelClaim = context.User.FindFirst("AccessLevel")
                                                     ?.Value;

                       if (accessLevelClaim == null || !byte.TryParse(accessLevelClaim, out var accessLevel))
                           return false;

                       return accessLevel == 2; // Allow admins only
                   }
               )
       )
       .AddPolicy
       (
           "ModeratorOrHigher",
           policy =>
               policy.RequireAssertion
               (context =>
                   {
                       var accessLevelClaim = context.User.FindFirst("AccessLevel")
                                                     ?.Value;

                       if (accessLevelClaim == null || !byte.TryParse(accessLevelClaim, out var accessLevel))
                           return false;

                       return accessLevel >= 1; // Allow moderators and higher
                   }
               )
       )
       .AddPolicy
       (
           "All",
           policy =>
               policy.RequireAssertion(context => true)
       );

var app = builder.Build();

// Ensure the database is created and migrations are applied
using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    var context  = services.GetRequiredService<ManoloDbContext>();

    // Apply any pending migrations
    context.Database.EnsureCreated();

    // After migrations, create dynamic tables
    var dynamicTableService = services.GetRequiredService<DynamicTableService>();
    dynamicTableService.CreateDynamicTables();

    // After migrations, seed the database with the default admin
    var adminSeeder = services.GetRequiredService<UserSeeder>();
    adminSeeder.Seed();
}

app.UsePathBase("/manolo-data-tier");

app.MapHub<SignalRHub>("/signalr");

app.UseSwagger();
app.UseSwaggerUI(c => { c.SwaggerEndpoint("/manolo-data-tier/swagger/v1/swagger.json", "API V1"); });

app.UseRouting();

app.Use
((context, next) =>
    {
        context.Response.Headers.AltSvc = $"""h3="{host}:{port}"; ma=86400""";

        return next();
    }
);

app.UseAuthentication();
app.UseAuthorization();

app.MapControllers();

app.Run();