using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Alias.DeleteAlias;

public class DeleteAliasHandler : IRequestHandler<DeleteAliasQuery, Result>{

    private readonly ManoloDbContext _context;


    public DeleteAliasHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(DeleteAliasQuery request,
                                     CancellationToken cancellationToken){

        var rowsAffected = await _context.Alias
                                         .Where(d => d.AliasName == request.Alias)
                                         .ExecuteDeleteAsync(cancellationToken);

        return rowsAffected == 0
            ? Result.Failure(DomainError.AliasDoesNotExist(request.Alias))
            : Result.Success();
    }

}