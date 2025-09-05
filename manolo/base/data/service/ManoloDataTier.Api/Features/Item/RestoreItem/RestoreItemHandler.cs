using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.Features.Item.RestoreItem;

public class RestoreItemHandler : IRequestHandler<RestoreItemQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public RestoreItemHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(RestoreItemQuery request, CancellationToken cancellationToken){

        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!existingDataStructure)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        var id = await _idResolverService.GetIdFromRequestAsync(request.Id, cancellationToken);

        var tableName = $"ItemDSN{request.Dsn}";

        var sql = $"""
            UPDATE "{tableName}" 
            SET "IsDeletedRaw" = 0
            WHERE "Id" = @Id;
            """;

        var parameters = new[]{
            new NpgsqlParameter("@Id", id),
        };

        await _context.Database.ExecuteSqlRawAsync(sql, parameters, cancellationToken);

        return Result.Success($"Item: {id} restored.");
    }

}