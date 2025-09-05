using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using MediatR;
using Microsoft.EntityFrameworkCore;
using KeyValueModel = ManoloDataTier.Storage.Model.KeyValue;

namespace ManoloDataTier.Api.Features.KeyValue.CreateUpdateKeyValue;

public class CreateUpdateKeyValueHandler : IRequestHandler<CreateUpdateKeyValueQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public CreateUpdateKeyValueHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(CreateUpdateKeyValueQuery request, CancellationToken cancellationToken){

        var objectId = await _idResolverService.GetIdFromRequestAsync(request.Object, cancellationToken);

        var count = 0;

        if (request.Key.Contains("edge_"))
            count = await _context.KeyValues
                                  .CountAsync(d => d.Key.Contains("edge_"), cancellationToken);

        count++;

        var existingKey = await _context.KeyValues
                                        .Where(d => d.Key == request.Key && d.Object == objectId)
                                        .FirstOrDefaultAsync(cancellationToken);

        if (existingKey != null){

            existingKey.Value              = request.Value;
            existingKey.LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        }
        else{

            var keyValue = new KeyValueModel{
                Id                 = KeyValueModel.GenerateId(),
                Object             = objectId,
                Key                = request.Key == "edge_" ? $"edge_{count}" : request.Key,
                Value              = request.Value,
                LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            };

            await _context.KeyValues.AddAsync(keyValue, cancellationToken);
        }

        await _context.SaveChangesAsync(cancellationToken);

        return Result.Success();
    }

}