using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.KeyValue.DeleteKeyValueBatch;

public class DeleteKeyValueBatchHandler : IRequestHandler<DeleteKeyValueBatchQuery, Result>{

    private readonly ManoloDbContext _context;

    public DeleteKeyValueBatchHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(DeleteKeyValueBatchQuery request, CancellationToken cancellationToken){

        if (request.Keys.Length == 0)
            return Result.Failure(DomainError.KeyValueBatchEmpty());

        var rowsAffected = await _context.KeyValues
                                         .Where(kv => kv.Object == request.Object && request.Keys.Contains(kv.Key))
                                         .ExecuteDeleteAsync(cancellationToken);

        return rowsAffected == 0
            ? Result.Failure(DomainError.KeyValueBatchDoesNotExist(request.Object, request.Keys))
            : Result.Success();
    }

}