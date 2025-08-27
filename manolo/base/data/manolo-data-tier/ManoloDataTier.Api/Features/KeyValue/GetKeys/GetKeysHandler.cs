using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.KeyValue.GetKeys;

public class GetKeysHandler : IRequestHandler<GetKeysQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public GetKeysHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(GetKeysQuery request,
                                     CancellationToken cancellationToken){

        var objectId = await _idResolverService.GetIdFromRequestAsync(request.Object, cancellationToken);

        var existingKeys = await _context.KeyValues
                                         .AsNoTracking()
                                         .Where(d => d.Object == objectId)
                                         .Select(d => d.Key)
                                         .ToListAsync(cancellationToken);

        return Result.Success(existingKeys);
    }

}