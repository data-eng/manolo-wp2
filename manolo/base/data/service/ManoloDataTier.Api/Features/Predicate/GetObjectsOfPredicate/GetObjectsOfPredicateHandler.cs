using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Predicate.GetObjectsOfPredicate;

public class GetObjectsOfPredicateHandler : IRequestHandler<GetObjectsOfPredicateQuery, Result>{

    private readonly ManoloDbContext _context;


    public GetObjectsOfPredicateHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(GetObjectsOfPredicateQuery request,
                                     CancellationToken cancellationToken){

        var existingPredicate = await _context.Predicates
                                              .AnyAsync(d => d.Description == request.Description, cancellationToken);

        if (!existingPredicate)
            Result.Failure(DomainError.NoPredicateExist(request.Description));

        var objects = await _context.Relations
                                    .Where(r => r.Predicate == request.Description)
                                    .Select(r => r.Object)
                                    .ToListAsync(cancellationToken);

        return objects.Count == 0
            ? Result.Failure(DomainError.NoRelations())
            : Result.Success(objects);

    }

}