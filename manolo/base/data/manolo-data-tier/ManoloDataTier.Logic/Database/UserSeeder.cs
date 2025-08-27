using ManoloDataTier.Logic.Extensions;
using ManoloDataTier.Storage.Model;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Logic.Database;

public class UserSeeder{

    private readonly ManoloDbContext _dbContext;
    private const    int             CurrentVersion = 1;

    public UserSeeder(ManoloDbContext dbContext){
        _dbContext = dbContext;
    }

    public void Seed(){
        var existing = _dbContext.SeedHistories.AsNoTracking()
                                 .FirstOrDefault(x => x.Version == CurrentVersion);

        if (existing is not null && existing.Version >= CurrentVersion)
            return;

        SeedAdmin();
        SeedPredicates();
        SeedDataStructures();
        _dbContext.SaveChanges();
    }

    private void SeedAdmin(){
        var existingUser = _dbContext.Users
                                     .AsNoTracking()
                                     .FirstOrDefault(u => u.Username == "manolo");

        if (existingUser != null){
            return;
        }

        var user = new User{
            Username     = "manolo",
            Id           = User.GenerateId(),
            PasswordHash = BCrypt.Net.BCrypt.HashPassword("manolo")!,
            AccessLevel  = 2,
        };

        _dbContext.Users.Add(user);
    }

    private void SeedPredicates(){
        var requiredDescriptions = new[]{
            "|_", "--", "->", "runs_on",
        };

        var existingDescriptions = _dbContext.Predicates
                                             .AsNoTracking()
                                             .Where(p => requiredDescriptions.Contains(p.Description))
                                             .Select(p => p.Description)
                                             .ToHashSet();

        foreach (var description in requiredDescriptions){
            if (!existingDescriptions.Contains(description)){
                _dbContext.Predicates.Add(new(){
                    Id                 = Predicate.GenerateId(),
                    Description        = description,
                    LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                });
            }
        }
    }

    private void SeedDataStructures(){
        List<Dictionary<string, int>> dataStructures =[
            new(){
                {
                    "mlflow", 1
                },{
                    "topology", 2
                },
            },
        ];

        var existingDataStructures = _dbContext.DataStructures
                                               .AsNoTracking()
                                               .Select(d => d.Dsn)
                                               .ToHashSet();

        foreach (var kvp in dataStructures.SelectMany(dict =>
                                                          dict.Where(kvp => !existingDataStructures.Contains(kvp.Value)))){
            _dbContext.DataStructures.Add(new(){
                Id                 = DataStructure.GenerateId(),
                Dsn                = kvp.Value,
                Name               = kvp.Key,
                Kind               = kvp.Key,
                LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            });
        }

        _dbContext.SaveChanges();

        var dynamicTableService = new DynamicTableService(_dbContext);
        dynamicTableService.CreateDynamicTables();
    }

}