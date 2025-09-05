namespace ManoloDataTier.Logic.Interfaces;

public interface IIdResolverService{

    Task<string> GetIdFromRequestAsync(string? requestId, CancellationToken cancellationToken);
    Task<string> GetAliasFromRequestAsync(string? requestId, CancellationToken cancellationToken);

}