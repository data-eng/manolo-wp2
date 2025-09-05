using ManoloDataTier.Logic.Database;
using Microsoft.EntityFrameworkCore;
using Npgsql;
using ItemModel = ManoloDataTier.Storage.Model.Item;

namespace ManoloDataTier.Logic.Services;

public class ServiceHelpers{

    private readonly ManoloDbContext _context;

    public ServiceHelpers(ManoloDbContext context){
        _context = context;
    }

    public async Task<bool> ValidateEntity(int dsn, string id, string entityType, CancellationToken cancellationToken){
        if (entityType == "ITS")
            return await GetItem(dsn, id, cancellationToken);

        return await _context.DataStructures
                             .AnyAsync(d => d.Id == id, cancellationToken);
    }

    private async Task<bool> GetItem(int dsn, string id, CancellationToken cancellationToken){
        var tableName = $"ItemDSN{dsn}";

        var sql = $"""
            SELECT *
            FROM "{tableName}" 
            WHERE "Id" = @Id
            """;

        object[] parameters =[
            new NpgsqlParameter("@Id", id),
        ];

        return await _context.Database
                             .SqlQueryRaw<ItemModel>(sql, parameters)
                             .AsNoTracking()
                             .AnyAsync(cancellationToken);
    }

}