using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.KeyValue.GetKeyValuePerObject;

public class GetKeyValuePerObjectHandler : IRequestHandler<GetKeyValuePerObjectQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public GetKeyValuePerObjectHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(GetKeyValuePerObjectQuery request,
                                     CancellationToken cancellationToken){

        var objectId = await _idResolverService.GetIdFromRequestAsync(request.Obj, cancellationToken);

        var kvps = await _context.KeyValues
                                 .AsNoTracking()
                                 .Where(d => d.Object == objectId)
                                 .ToListAsync(cancellationToken);

        return Result.Success(kvps);
    }

}