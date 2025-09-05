using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;
using KeyValueModel = ManoloDataTier.Storage.Model.KeyValue;

namespace ManoloDataTier.Api.Features.KeyValue.CreateUpdateKeyValueBatch;

public class CreateUpdateKeyValueBatchHandler : IRequestHandler<CreateUpdateKeyValueBatchQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;

    public CreateUpdateKeyValueBatchHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(CreateUpdateKeyValueBatchQuery request, CancellationToken cancellationToken){

        if (request.Keys.Length != request.Values.Length)
            return Result.Failure(DomainError.KeyValueBatchMismatch(request.Keys.Length, request.Values.Length));

        var objectId = await _idResolverService.GetIdFromRequestAsync(request.Object, cancellationToken);

        var keys   = request.Keys;
        var values = request.Values;

        var existingKeyValues = await _context.KeyValues
                                              .Where(kv => kv.Object == objectId && keys.Contains(kv.Key))
                                              .ToListAsync(cancellationToken);

        var existingDict = existingKeyValues.ToDictionary(kv => kv.Key, kv => kv);

        var edgeCount = await _context.KeyValues
                                      .AsNoTracking()
                                      .CountAsync(kv => kv.Object == objectId && kv.Key.StartsWith("edge_"), cancellationToken);

        var now          = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var newKeyValues = new List<KeyValueModel>();

        for (var i = 0; i < keys.Length; i++){
            var key   = keys[i];
            var value = values[i];

            if (existingDict.TryGetValue(key, out var existing)){

                existing.Value              = value;
                existing.LastChangeDateTime = now;
            }
            else{

                var finalKey = key == "edge_" ? $"edge_{edgeCount++}" : key;

                newKeyValues.Add(new(){
                    Id                 = KeyValueModel.GenerateId(),
                    Object             = objectId,
                    Key                = finalKey,
                    Value              = value,
                    LastChangeDateTime = now,
                });
            }
        }

        if (newKeyValues.Count > 0)
            await _context.KeyValues.AddRangeAsync(newKeyValues, cancellationToken);

        await _context.SaveChangesAsync(cancellationToken);

        return Result.Success();
    }

}