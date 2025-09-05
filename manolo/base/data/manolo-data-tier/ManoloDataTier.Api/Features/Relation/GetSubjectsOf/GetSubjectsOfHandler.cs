using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Relation.GetSubjectsOf;

public class GetSubjectsOfHandler : IRequestHandler<GetSubjectsOfQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public GetSubjectsOfHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(GetSubjectsOfQuery request,
                                     CancellationToken cancellationToken){

        var existingPredicate = await _context.Predicates
                                              .AnyAsync(d => d.Description == request.Description, cancellationToken);

        if (!existingPredicate)
            Result.Failure(DomainError.NoPredicatesExist());

        var objId = await _idResolverService.GetIdFromRequestAsync(request.Object, cancellationToken);

        IQueryable<string> query;

        if (request.Description == "getAdjacencyList"){

            query = _context.Relations
                            .Where(relation =>
                                       ( relation.Predicate == "--" || relation.Predicate == "->" ) && relation.Object == objId)
                            .Select(relation => relation.Subject);
        }
        else{

            query = _context.Relations
                            .Where(relation => relation.Predicate == request.Description && relation.Object == objId)
                            .Select(relation => relation.Subject);
        }

        var relations = await query
                              .AsNoTracking()
                              .ToListAsync(cancellationToken);

        return relations.Count == 0
            ? Result.Failure(DomainError.NoRelations())
            : Result.Success(relations);
    }

}