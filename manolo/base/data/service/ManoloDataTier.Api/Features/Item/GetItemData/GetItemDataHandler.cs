using System.Data;
using System.Text;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using ManoloDataTier.Logic.Services;
using ManoloDataTier.Storage.Dto;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.Features.Item.GetItemData;

public class GetItemDataHandler : IRequestHandler<GetItemDataQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;
    private readonly MlflowService      _mlflowService;

    public GetItemDataHandler(ManoloDbContext context, IIdResolverService idResolverService,
                              MlflowService mlflowService){
        _context           = context;
        _idResolverService = idResolverService;
        _mlflowService     = mlflowService;
    }

    public async Task<Result> Handle(GetItemDataQuery request, CancellationToken cancellationToken){

        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!existingDataStructure)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        var alias = await _idResolverService.GetAliasFromRequestAsync(request.Id, cancellationToken);

        if (alias != string.Empty)
            if (alias.Contains("mlflow", StringComparison.InvariantCultureIgnoreCase))
                return Result.Success(await _mlflowService.GetDataByEntityAsync(alias));

        var id = await _idResolverService.GetIdFromRequestAsync(request.Id, cancellationToken);

        var tableName = $"ItemDSN{request.Dsn}";

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
                                 .FirstOrDefaultAsync(cancellationToken);

        if (item == null)
            return Result.Failure(DomainError.ItemDataIsFile());

        byte[] data;

        await using (var conn = _context.Database.GetDbConnection() as NpgsqlConnection){
            if (conn == null)
                return Result.Failure(DomainError.DatabaseConnectionError());

            if (conn.State != ConnectionState.Open)
                await conn.OpenAsync(cancellationToken);

            await using var transaction = await conn.BeginTransactionAsync(cancellationToken);

            data = await DatabaseHelpers.ReadLargeObjectAsync(conn, transaction, item.DataOid, cancellationToken);

            await transaction.CommitAsync(cancellationToken);
        }

        var decodedString = Encoding.UTF8.GetString(data);

        return Result.Success(decodedString);
    }

}