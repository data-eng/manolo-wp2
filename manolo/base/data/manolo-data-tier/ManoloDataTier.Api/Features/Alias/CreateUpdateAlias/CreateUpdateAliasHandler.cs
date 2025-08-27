using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;
using AliasModel = ManoloDataTier.Storage.Model.Alias;

namespace ManoloDataTier.Api.Features.Alias.CreateUpdateAlias;

public class CreateUpdateAliasHandler : IRequestHandler<CreateUpdateAliasQuery, Result>{

    private readonly ManoloDbContext _context;


    public CreateUpdateAliasHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(CreateUpdateAliasQuery request,
                                     CancellationToken cancellationToken){

        var existingAlias = await _context.Alias
                                          .AsNoTracking()
                                          .CountAsync(d => d.Id != request.Id && d.AliasName == request.Alias, cancellationToken);

        if (existingAlias != 0)
            return Result.Failure(DomainError.AliasAlreadyExistsId(request.Alias));

        var alias = new AliasModel{
            Id        = request.Id,
            AliasName = request.Alias,
        };

        await _context.Alias.AddAsync(alias, cancellationToken);

        await _context.SaveChangesAsync(cancellationToken);

        return Result.Success();
    }

}