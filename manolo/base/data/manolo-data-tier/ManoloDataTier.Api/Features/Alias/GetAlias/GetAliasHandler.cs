using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Alias.GetAlias;

public class GetAliasHandler : IRequestHandler<GetAliasQuery, Result>{

    private readonly ManoloDbContext _context;


    public GetAliasHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(GetAliasQuery request,
                                     CancellationToken cancellationToken){

        var existingAlias = await _context.Alias
                                          .AsNoTracking()
                                          .FirstOrDefaultAsync(d => d.Id == request.Id, cancellationToken);

        return existingAlias != null
            ? Result.Success(existingAlias.AliasName)
            : Result.Failure(DomainError.IdDoesNotHaveAlias(request.Id));
    }

}