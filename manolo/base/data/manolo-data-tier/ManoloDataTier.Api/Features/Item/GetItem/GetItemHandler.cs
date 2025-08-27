using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using ManoloDataTier.Storage.Dto;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.Features.Item.GetItem;

public class GetItemHandler : IRequestHandler<GetItemQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public GetItemHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(GetItemQuery request, CancellationToken cancellationToken){

        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!existingDataStructure)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        var id = await _idResolverService.GetIdFromRequestAsync(request.Id, cancellationToken);

        var tableName = $"ItemDSN{request.Dsn}";

        var sql = $"""
            SELECT "Id", "ForeignDsn","MimeType", "IsForeignRaw", "LastChangeDateTime", "IsDeletedRaw", "IsFileRaw"
            FROM "{tableName}" 
            WHERE "Id" = @Id
            """;

        var parameter = new NpgsqlParameter("@Id", id);

        var item = await _context.Database
                                 .SqlQueryRaw<ItemDto>(sql, parameter)
                                 .AsNoTracking()
                                 .FirstOrDefaultAsync(cancellationToken);

        return Result.Success(item);
    }

}