using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.DataStructure.UpdateDataStructure;

public class UpdateDataStructureHandler : IRequestHandler<UpdateDataStructureQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public UpdateDataStructureHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(UpdateDataStructureQuery request,
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

        if (!string.IsNullOrEmpty(request.Name))
            existingDataStructure.Name = request.Name;

        if (request.Kind != null)
            existingDataStructure.Kind = request.Kind;

        existingDataStructure.LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        await _context.SaveChangesAsync(cancellationToken);

        return Result.Success("DataStructure Updated.");
    }

}