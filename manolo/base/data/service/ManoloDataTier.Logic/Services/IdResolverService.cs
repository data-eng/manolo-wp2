using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Interfaces;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Logic.Services;

public class IdResolverService : IIdResolverService{

    private readonly ManoloDbContext _context;

    public IdResolverService(ManoloDbContext context){
        _context = context;
    }

    public async Task<string> GetIdFromRequestAsync(string? requestId, CancellationToken cancellationToken){
        if (string.IsNullOrEmpty(requestId))
            return string.Empty;

        return requestId.Length == 29
            ? requestId
            : await _context.Alias
                            .AsNoTracking()
                            .Where(d => d.AliasName == requestId)
                            .Select(d => d.Id)
                            .FirstOrDefaultAsync(cancellationToken)
           ?? requestId;
    }

    public async Task<string> GetAliasFromRequestAsync(string? requestId, CancellationToken cancellationToken){
        if (string.IsNullOrEmpty(requestId))
            return string.Empty;

        var alias = await _context.Alias
                                  .AsNoTracking()
                                  .Where(d => d.Id == requestId)
                                  .Select(d => d.AliasName)
                                  .FirstOrDefaultAsync(cancellationToken);

        return alias ?? requestId;
    }

}