using System.Data;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Interfaces;
using ManoloDataTier.Logic.Services;
using ManoloDataTier.Storage.Dto;
using MediatR;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.Features.Item.DownloadItemData;

public class DownloadItemDataHandler : IRequestHandler<DownloadItemDataQuery, IActionResult>
{

    private readonly ManoloDbContext _context;

    private readonly IIdResolverService _idResolverService;

    public DownloadItemDataHandler(ManoloDbContext context, IIdResolverService idResolverService)
    {
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<IActionResult> Handle(DownloadItemDataQuery request, CancellationToken cancellationToken)
    {
        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!existingDataStructure)
            return new BadRequestResult();

        var id = await _idResolverService.GetIdFromRequestAsync(request.Id, cancellationToken);

        var tableName = $"ItemDSN{request.Dsn}";

        var sql = $"""
                SELECT * 
                FROM "{tableName}" 
                WHERE "Id" = @Id
            """;

        ItemDto item;

        if (_context.Database.GetDbConnection() is not NpgsqlConnection conn)
            return new NotFoundResult();

        if (conn.State != ConnectionState.Open)
            await conn.OpenAsync(cancellationToken);

        await using (var cmd = new NpgsqlCommand(sql, conn))
        {
            cmd.Parameters.AddWithValue("@Id", id);
            await using var reader = await cmd.ExecuteReaderAsync(cancellationToken);

            if (await reader.ReadAsync(cancellationToken))
            {
                item = new()
                {
                    Id                 = reader.GetString(reader.GetOrdinal("Id")),
                    ForeignDsn         = reader.GetInt32(reader.GetOrdinal("ForeignDsn")),
                    MimeType           = reader.GetString(reader.GetOrdinal("MimeType")),
                    IsForeignRaw       = reader.GetInt32(reader.GetOrdinal("IsForeignRaw")),
                    LastChangeDateTime = reader.GetInt64(reader.GetOrdinal("LastChangeDateTime")),
                    IsDeletedRaw       = reader.GetInt32(reader.GetOrdinal("IsDeletedRaw")),
                    IsFileRaw          = reader.GetInt32(reader.GetOrdinal("IsFileRaw")),
                    DataOid            = (uint)reader.GetInt64(reader.GetOrdinal("DataOid")),
                };
            }
            else
            {
                return new NotFoundResult();
            }
        }

        if (string.IsNullOrEmpty(item.Id) || item.IsFileRaw == 0)
            return new NotFoundResult();

        byte[] data;

        await using (var transaction = await conn.BeginTransactionAsync(cancellationToken))
        {
            data = await DatabaseHelpers.ReadLargeObjectAsync(conn, transaction, item.DataOid, cancellationToken);
            await transaction.CommitAsync(cancellationToken);
        }

        var fileExtension = item.MimeType;
        var fileName      = $"Item_{id}.{fileExtension}";

        return new FileContentResult(data, "application/octet-stream")
        {
            FileDownloadName = fileName,
        };
    }

}