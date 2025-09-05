using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.Features.Item.DeleteItem;

public class DeleteItemHandler : IRequestHandler<DeleteItemQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public DeleteItemHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(DeleteItemQuery request, CancellationToken cancellationToken){

        var existingDataStructures = await _context.DataStructures
                                                   .AsNoTracking()
                                                   .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        var id = await _idResolverService.GetIdFromRequestAsync(request.Id, cancellationToken);

        if (!existingDataStructures)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        var tableName = $"ItemDSN{request.Dsn}";

        var sql = $"""
            UPDATE "{tableName}" 
            SET "IsDeletedRaw" = 1
            WHERE "Id" = @Id;
            """;

        var parameters = new[]{
            new NpgsqlParameter("@Id", id),
        };

        await _context.Database.ExecuteSqlRawAsync(sql, parameters, cancellationToken);

        return Result.Success($"Item: {id} deleted.");
    }

}