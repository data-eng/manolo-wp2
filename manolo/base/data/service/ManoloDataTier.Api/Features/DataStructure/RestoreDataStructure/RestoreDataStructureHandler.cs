using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.DataStructure.RestoreDataStructure;

public class RestoreDataStructureHandler : IRequestHandler<RestoreDataStructureQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public RestoreDataStructureHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(RestoreDataStructureQuery request,
                                     CancellationToken cancellationToken){
        var id = await _idResolverService.GetIdFromRequestAsync(request.Id, cancellationToken);

        if (string.IsNullOrEmpty(id) && string.IsNullOrEmpty(request.Name) && request.Dsn == -1)
            return Result.Failure(DomainError.NoQueryProvided());

        var query = _context.DataStructures
                            .AsQueryable();

        if (!string.IsNullOrEmpty(id))
            query = query.Where(d => d.Id == id);

        if (!string.IsNullOrEmpty(request.Name))
            query = query.Where(d => d.Name == request.Name);

        if (request.Dsn != -1)
            query = query.Where(d => d.Dsn == request.Dsn);

        var existingDataStructure = await query
            .FirstOrDefaultAsync(cancellationToken: cancellationToken);

        if (existingDataStructure is null){

            if (!string.IsNullOrEmpty(id))
                return Result.Failure(DomainError.DataStructureDoesNotExistId(id));

            if (!string.IsNullOrEmpty(request.Name))
                return Result.Failure(DomainError.DataStructureDoesNotExistName(request.Name));

            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));
        }

        existingDataStructure.IsDeletedRaw = 0;

        await _context.SaveChangesAsync(cancellationToken);

        return Result.Success($"Successfully deleted DataStructure {existingDataStructure.Name}.");
    }

}