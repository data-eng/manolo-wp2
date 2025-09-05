using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Predicate.DeletePredicate;

public class DeletePredicateHandler : IRequestHandler<DeletePredicateQuery, Result>{

    private readonly ManoloDbContext _context;


    public DeletePredicateHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(DeletePredicateQuery request,
                                     CancellationToken cancellationToken){

        var rowsAffected = await _context.Predicates
                                         .Where(d => d.Description == request.Description)
                                         .ExecuteDeleteAsync(cancellationToken);

        return rowsAffected == 0 ? Result.Failure(DomainError.NoPredicateExist(request.Description)) : Result.Success();
    }

}