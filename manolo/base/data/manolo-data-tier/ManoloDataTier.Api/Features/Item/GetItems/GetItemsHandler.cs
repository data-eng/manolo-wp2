using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Item.GetItems;

public class GetItemsHandler : IRequestHandler<GetItemsQuery, Result>{

    private readonly ManoloDbContext _context;


    public GetItemsHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(GetItemsQuery request, CancellationToken cancellationToken){
        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!existingDataStructure)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        var tableName = $"ItemDSN{request.Dsn}";

        var sql = $"""
            SELECT "Id"
            FROM "{tableName}"
            WHERE "IsDeletedRaw" = 0 
            """;

        var itemsList = await _context.Database
                                      .SqlQueryRaw<string>(sql)
                                      .AsNoTracking()
                                      .ToListAsync(cancellationToken);

        return Result.Success(itemsList);
    }

}