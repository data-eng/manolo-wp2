using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.KeyValue.GetValue;

public class GetValueHandler : IRequestHandler<GetValueQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public GetValueHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(GetValueQuery request,
                                     CancellationToken cancellationToken){

        var objectId = await _idResolverService.GetIdFromRequestAsync(request.Object, cancellationToken);

        var existingKeys = await _context.KeyValues
                                         .AsNoTracking()
                                         .Where(d => d.Key == request.Key && d.Object == objectId)
                                         .ToListAsync(cancellationToken);

        return Result.Success(existingKeys);
    }

}