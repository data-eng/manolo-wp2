using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Alias.GetId;

public class GetIdHandler : IRequestHandler<GetIdQuery, Result>{

    private readonly ManoloDbContext _context;


    public GetIdHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(GetIdQuery request,
                                     CancellationToken cancellationToken){

        var existingAlias = await _context.Alias
                                          .AsNoTracking()
                                          .FirstOrDefaultAsync(d => d.AliasName == request.Alias, cancellationToken);

        return existingAlias == null
            ? Result.Failure(DomainError.AliasDoesNotExist(request.Alias))
            : Result.Success(existingAlias.Id);
    }

}