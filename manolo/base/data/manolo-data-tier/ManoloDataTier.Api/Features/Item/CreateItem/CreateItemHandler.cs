using System.Data;
using System.Text;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Services;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.Features.Item.CreateItem;

public class CreateItemHandler : IRequestHandler<CreateItemQuery, Result>{

    private readonly ManoloDbContext _context;

    public CreateItemHandler(ManoloDbContext context){
        _context = context;
    }


    public async Task<Result> Handle(CreateItemQuery request, CancellationToken cancellationToken){

        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!existingDataStructure)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        Stream dataStream;
        var    mimeType = "txt";
        var    isFile   = 0;

        if (!string.IsNullOrEmpty(request.Data)){
            dataStream = new MemoryStream(Encoding.UTF8.GetBytes(request.Data));
        }
        else if (request.DataFile?.Length > 0){
            dataStream = request.DataFile.OpenReadStream();
            isFile     = 1;

            var lastDot = request.DataFile.FileName.LastIndexOf('.');

            if (lastDot >= 0 && lastDot < request.DataFile.FileName.Length - 1)
                mimeType = request.DataFile.FileName[( lastDot + 1 )..];
        }
        else{
            return Result.Failure(DomainError.NoDataProvided());
        }

        await using var conn = _context.Database.GetDbConnection() as NpgsqlConnection;

        if (conn == null)
            return Result.Failure(DomainError.DatabaseConnectionError());

        if (conn.State != ConnectionState.Open)
            await conn.OpenAsync(cancellationToken);

        await using var transaction = await conn.BeginTransactionAsync(cancellationToken);

        var insertedIds = new List<string>();
        await DatabaseHelpers.ProcessSingleItem(conn, transaction, request.Dsn, dataStream, mimeType, isFile, insertedIds, cancellationToken);

        await transaction.CommitAsync(cancellationToken);

        return Result.Success(insertedIds.First());
    }

}