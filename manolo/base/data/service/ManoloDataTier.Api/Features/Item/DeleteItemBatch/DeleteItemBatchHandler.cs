using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Npgsql;
using NpgsqlTypes;

namespace ManoloDataTier.Api.Features.Item.DeleteItemBatch;

public class DeleteItemBatchHandler : IRequestHandler<DeleteItemBatchQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;

    public DeleteItemBatchHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(DeleteItemBatchQuery request, CancellationToken cancellationToken){

        var idTasks = request.Ids
                             .Select(idStr => _idResolverService
                                         .GetIdFromRequestAsync(idStr, cancellationToken));

        var ids = ( await Task.WhenAll(idTasks) ).ToList();

        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!existingDataStructure)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        var tableName = $"ItemDSN{request.Dsn}";

        var sql = $"""
            UPDATE "{tableName}"
            SET "IsDeletedRaw" = 1
            WHERE "Id" = ANY(@Ids);
            """;

        var parameter = new NpgsqlParameter("@Ids", ids.ToArray()){
            NpgsqlDbType = NpgsqlDbType.Array | NpgsqlDbType.Text,
        };

        await _context.Database.ExecuteSqlRawAsync(sql, [parameter,], cancellationToken);

        return Result.Success($"Items deleted: {ids.Count}");
    }

}