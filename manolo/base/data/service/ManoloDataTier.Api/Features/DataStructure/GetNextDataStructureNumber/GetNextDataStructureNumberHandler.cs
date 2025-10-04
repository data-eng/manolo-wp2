using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.DataStructure.GetNextDataStructureNumber;

public class GetNextDataStructureNumberHandler : IRequestHandler<GetNextDataStructureNumberQuery, Result>
{

    private readonly ManoloDbContext _context;


    public GetNextDataStructureNumberHandler(ManoloDbContext context)
    {
        _context = context;
    }

    public async Task<Result> Handle(
        GetNextDataStructureNumberQuery request,
        CancellationToken               cancellationToken
    )
    {
        var usedNumbers = await _context.DataStructures
                                        .AsNoTracking()
                                        .Where(d => d.IsDeletedRaw == 0)
                                        .Select(d => d.Dsn)
                                        .ToListAsync(cancellationToken);

        var nextAvailable = 11;

        if (usedNumbers.Count != 0)
        {
            usedNumbers.Sort();

            foreach (var num in usedNumbers)
            {
                if (num == nextAvailable)
                    nextAvailable++;

                else if (num > nextAvailable)
                    break;
            }
        }

        return Result.Success(nextAvailable);
    }

}