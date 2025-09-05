using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Relation.GetRelations;

public class GetRelationsHandler : IRequestHandler<GetRelationsQuery, Result>{

    private readonly ManoloDbContext _context;

    public GetRelationsHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(GetRelationsQuery request,
                                     CancellationToken cancellationToken){
        var relations = await _context.Relations
                                      .AsNoTracking()
                                      .ToListAsync(cancellationToken);

        return relations.Count == 0 ? Result.Failure(DomainError.NoRelations()) : Result.Success(relations);
    }

}