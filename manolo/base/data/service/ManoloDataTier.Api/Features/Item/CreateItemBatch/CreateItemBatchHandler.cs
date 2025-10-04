using System.Data;
using System.Text;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Services;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.Features.Item.CreateItemBatch;

public class CreateItemBatchHandler : IRequestHandler<CreateItemBatchQuery, Result>
{

    private readonly ManoloDbContext _context;

    public CreateItemBatchHandler(ManoloDbContext context)
    {
        _context = context;
    }

    public async Task<Result> Handle(CreateItemBatchQuery request, CancellationToken cancellationToken)
    {
        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!existingDataStructure)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        if (( request.Data?.Length ?? 0 ) + ( request.DataFiles?.Length ?? 0 ) == 0)
            return Result.Failure(DomainError.NoDataProvided());

        var insertedIds = new List<string>(); //check if it works

        await using var conn = _context.Database.GetDbConnection() as NpgsqlConnection;

        if (conn == null)
            return Result.Failure(DomainError.DatabaseConnectionError());

        if (conn.State != ConnectionState.Open)
            await conn.OpenAsync(cancellationToken);

        await using var transaction = await conn.BeginTransactionAsync(cancellationToken);

        if (request.Data != null)
            foreach (var text in request.Data)
            {
                if (string.IsNullOrEmpty(text))
                    continue;

                await DatabaseHelpers.ProcessSingleItem
                (
                    conn,
                    transaction,
                    request.Dsn,
                    new MemoryStream(Encoding.UTF8.GetBytes(text)),
                    "txt",
                    0,
                    insertedIds,
                    cancellationToken
                );
            }

        if (request.DataFiles != null)
            foreach (var file in request.DataFiles)
            {
                if (file.Length == 0)
                    continue;

                var lastDot = file.FileName.LastIndexOf('.');

                var mimeType = lastDot >= 0 && lastDot < file.FileName.Length - 1
                    ? file.FileName[( lastDot + 1 )..]
                    : "bin";

                await using var fileStream = file.OpenReadStream();
                await DatabaseHelpers.ProcessSingleItem(conn, transaction, request.Dsn, fileStream, mimeType, 1, insertedIds, cancellationToken);
            }

        await transaction.CommitAsync(cancellationToken);

        return Result.Success
        (
            insertedIds.Count == 1
                ? insertedIds[0]
                : insertedIds.ToArray()
        );
    }

}