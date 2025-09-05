using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.DataStructure.GetDataStructures;

public class GetDataStructuresHandler : IRequestHandler<GetDataStructuresQuery, Result>{

    private readonly ManoloDbContext _context;


    public GetDataStructuresHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(GetDataStructuresQuery request,
                                     CancellationToken cancellationToken){
        var existingDataStructures = await _context.DataStructures
                                                   .AsNoTracking()
                                                   .Where(d => d.IsDeletedRaw == 0)
                                                   .ToListAsync(cancellationToken);

        return Result.Success(existingDataStructures);
    }

}