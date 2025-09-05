using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Predicate.GetSubjectsOfPredicate;

public class GetSubjectsOfPredicateHandler : IRequestHandler<GetSubjectsOfPredicateQuery, Result>{

    private readonly ManoloDbContext _context;


    public GetSubjectsOfPredicateHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(GetSubjectsOfPredicateQuery request,
                                     CancellationToken cancellationToken){

        var existingPredicate = await _context.Predicates
                                              .AnyAsync(d => d.Description == request.Description, cancellationToken);

        if (!existingPredicate)
            return Result.Failure(DomainError.NoPredicateExist(request.Description));

        var subjects = await _context.Relations
                                     .Where(r => r.Predicate == request.Description)
                                     .Select(r => r.Subject)
                                     .ToListAsync(cancellationToken);

        return subjects.Count == 0
            ? Result.Failure(DomainError.NoRelations())
            : Result.Success(subjects);

    }

}