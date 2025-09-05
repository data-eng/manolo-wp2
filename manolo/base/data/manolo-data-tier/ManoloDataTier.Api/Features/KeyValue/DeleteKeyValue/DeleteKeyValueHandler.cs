using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.KeyValue.DeleteKeyValue;

public class DeleteKeyValueHandler : IRequestHandler<DeleteKeyValueQuery, Result>{

    private readonly ManoloDbContext _context;


    public DeleteKeyValueHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(DeleteKeyValueQuery request,
                                     CancellationToken cancellationToken){

        var rowsAffected = await _context.KeyValues
                                         .Where(d => d.Key == request.Key && d.Object == request.Object)
                                         .ExecuteDeleteAsync(cancellationToken);

        return rowsAffected == 0
            ? Result.Failure(DomainError.KeyValueDoesNotExist(request.Key))
            : Result.Success();
    }

}