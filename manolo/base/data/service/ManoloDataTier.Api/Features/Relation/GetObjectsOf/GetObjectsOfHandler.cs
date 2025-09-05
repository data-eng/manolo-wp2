using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Relation.GetObjectsOf;

public class GetObjectsOfHandler : IRequestHandler<GetObjectsOfQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public GetObjectsOfHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(GetObjectsOfQuery request,
                                     CancellationToken cancellationToken){

        var existingPredicate = await _context.Predicates
                                              .AnyAsync(d => d.Description == request.Description, cancellationToken);

        if (!existingPredicate)
            Result.Failure(DomainError.NoPredicateExist(request.Description));

        var subjId = await _idResolverService.GetIdFromRequestAsync(request.Subject, cancellationToken);

        IQueryable<string> query;

        if (request.Description == "getAdjacencyList"){

            query = _context.Relations
                            .Where(relation =>
                                       ( relation.Predicate == "--" || relation.Predicate == "->" ) && relation.Subject == subjId)
                            .Select(relation => relation.Object);
        }
        else{

            query = _context.Relations
                            .Where(relation => relation.Predicate == request.Description && relation.Subject == subjId)
                            .Select(relation => relation.Object);
        }

        var relations = await query
                              .AsNoTracking()
                              .ToListAsync(cancellationToken);

        return relations.Count == 0
            ? Result.Failure(DomainError.NoRelations())
            : Result.Success(relations);
    }

}