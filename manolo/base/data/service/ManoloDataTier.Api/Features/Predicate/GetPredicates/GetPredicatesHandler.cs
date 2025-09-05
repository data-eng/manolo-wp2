using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Predicate.GetPredicates;

public class GetPredicatesHandler : IRequestHandler<GetPredicatesQuery, Result>{

    private readonly ManoloDbContext _context;


    public GetPredicatesHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(GetPredicatesQuery request,
                                     CancellationToken cancellationToken){

        var existingPredicates = await _context.Predicates
                                               .ToListAsync(cancellationToken);

        return Result.Success(existingPredicates);
    }

}