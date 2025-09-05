using System.Data;
using System.Text;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using ManoloDataTier.Logic.Services;
using ManoloDataTier.Storage.Dto;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.SignalRFeatures.Item.GetItemData;

public class GetItemDataSignalR{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;
    private readonly MlflowService      _mlflowService;

    public GetItemDataSignalR(
        ManoloDbContext context,
        IIdResolverService idResolverService,
        MlflowService mlflowService){

        _context           = context;
        _idResolverService = idResolverService;
        _mlflowService     = mlflowService;
    }

    async internal Task GetItemDataAsync(IHubCallerClients clients, string connectionId, string requestId, int dsn){

        var existingDataStructures = await _context.DataStructures
                                                   .AnyAsync(d => d.Dsn == dsn);

        if (!existingDataStructures){

            await clients.Client(connectionId).SendAsync("SignalRError", DomainError.DataStructureDoesNotExistDsn(dsn));

            return;
        }

        var alias = await _idResolverService.GetAliasFromRequestAsync(requestId, CancellationToken.None);

        if (!string.IsNullOrEmpty(alias) && alias.Contains("mlflow", StringComparison.InvariantCultureIgnoreCase)){
            var mlflowData = await _mlflowService.GetDataByEntityAsync(alias);

            await clients.Client(connectionId).SendAsync("SignalRISuccess", mlflowData);

            return;
        }

        var id = await _idResolverService.GetIdFromRequestAsync(requestId, CancellationToken.None);

        var tableName = $"ItemDSN{dsn}";

        var sql = $"""
            SELECT "DataOid" 
            FROM "{tableName}"
            WHERE "Id" = @Id
            AND "IsFileRaw" = 0
            """;

        var parameter = new NpgsqlParameter("@Id", id);

        var item = await _context.Database
                                 .SqlQueryRaw<ItemOidDto>(sql, parameter)
                                 .AsNoTracking()
                                 .FirstOrDefaultAsync();

        if (item == null){

            await clients.Client(connectionId).SendAsync("SignalRError", DomainError.ItemDoesNotExist(id, dsn));

            return;
        }

        await using var conn = _context.Database.GetDbConnection() as NpgsqlConnection;

        if (conn == null){
            await clients.Client(connectionId).SendAsync("SignalRError", DomainError.DatabaseConnectionError());

            return;
        }

        if (conn.State != ConnectionState.Open)
            await conn.OpenAsync();

        await using var transaction = await conn.BeginTransactionAsync();

        var data = await DatabaseHelpers.ReadLargeObjectAsync(conn, transaction, item.DataOid, CancellationToken.None);

        await transaction.CommitAsync();

        var decodedString = Encoding.UTF8.GetString(data);

        await clients.Client(connectionId).SendAsync("SignalRISuccess", decodedString);

    }

}