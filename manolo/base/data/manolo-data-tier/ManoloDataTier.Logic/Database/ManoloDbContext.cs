using System.Reflection;
using ManoloDataTier.Storage.Model;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;

namespace ManoloDataTier.Logic.Database;

public class ManoloDbContext : DbContext{

    private readonly string? _schema;

    public ManoloDbContext(DbContextOptions<ManoloDbContext> options, IConfiguration configuration) : base(options){
        _schema = configuration.GetSection("DatabaseSettings")["Schema"];
    }

    public DbSet<DataStructure> DataStructures{ get; set; }
    public DbSet<User>          Users         { get; set; }
    public DbSet<Predicate>     Predicates    { get; set; }
    public DbSet<Relation>      Relations     { get; set; }
    public DbSet<Alias>         Alias         { get; set; }

    public DbSet<SeedHistory> SeedHistories{ get; set; }

    public DbSet<KeyValue> KeyValues{ get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder){
        base.OnModelCreating(modelBuilder);

        if (!string.IsNullOrWhiteSpace(_schema))
            modelBuilder.HasDefaultSchema(_schema);

        modelBuilder.ApplyConfigurationsFromAssembly(Assembly.GetExecutingAssembly());
    }

}