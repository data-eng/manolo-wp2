using System.Data;
using System.Text;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Services;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.Features.Item.CreateItemBatch;

public class CreateItemBatchHandler : IRequestHandler<CreateItemBatchQuery, Result>{

    private readonly ManoloDbContext _context;

    public CreateItemBatchHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(CreateItemBatchQuery request, CancellationToken cancellationToken){

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
            foreach (var text in request.Data){
                if (string.IsNullOrEmpty(text)) continue;

                await DatabaseHelpers.ProcessSingleItem(conn, transaction, request.Dsn, new MemoryStream(Encoding.UTF8.GetBytes(text)), "txt", 0,
                                                        insertedIds,
                                                        cancellationToken);
            }

        if (request.DataFiles != null)
            foreach (var file in request.DataFiles){
                if (file.Length == 0) continue;

                var lastDot = file.FileName.LastIndexOf('.');

                var mimeType = lastDot >= 0 && lastDot < file.FileName.Length - 1
                    ? file.FileName[( lastDot + 1 )..]
                    : "bin";

                await using var fileStream = file.OpenReadStream();
                await DatabaseHelpers.ProcessSingleItem(conn, transaction, request.Dsn, fileStream, mimeType, 1, insertedIds, cancellationToken);
            }

        await transaction.CommitAsync(cancellationToken);

        return Result.Success(insertedIds.Count == 1 ? insertedIds[0] : insertedIds.ToArray());
    }

    // public async Task<Result> Handle(CreateItemBatchQuery request, CancellationToken cancellationToken){
    //     var existingDataStructure = await _context.DataStructures
    //                                               .Where(d => d.Dsn == request.Dsn)
    //                                               .FirstOrDefaultAsync(cancellationToken);
    //
    //     if (existingDataStructure == null)
    //         return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));
    //
    //     var itemsToInsert = new List<ItemModel>();
    //
    //     if (request.Data != null){
    //         itemsToInsert.AddRange(request.Data
    //                                       .Select(dataStr => string.IsNullOrEmpty(dataStr) ?[] : Encoding.UTF8.GetBytes(dataStr))
    //                                       .Select(dataBytes => new ItemModel{
    //                                           Id                 = ItemModel.GenerateId(),
    //                                           Data               = dataBytes,
    //                                           ForeignDsn         = -1,
    //                                           MimeType           = "txt",
    //                                           IsForeignRaw       = 0,
    //                                           LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
    //                                           IsDeletedRaw       = 0,
    //                                           IsFileRaw          = 0,
    //                                       }));
    //     }
    //
    //     if (request.DataFiles != null){
    //         foreach (var file in request.DataFiles){
    //             if (file.Length == 0) continue;
    //
    //             byte[] fileData;
    //             await using (var memoryStream = new MemoryStream()){
    //                 await file.CopyToAsync(memoryStream, cancellationToken);
    //                 fileData = memoryStream.ToArray();
    //             }
    //
    //             var lastDotIndex = file.FileName.LastIndexOf('.');
    //             var mimeType     = lastDotIndex >= 0 ? file.FileName[( lastDotIndex + 1 )..] : "bin";
    //
    //             var newItem = new ItemModel{
    //                 Id                 = ItemModel.GenerateId(),
    //                 Data               = fileData,
    //                 ForeignDsn         = -1,
    //                 MimeType           = mimeType,
    //                 IsForeignRaw       = 0,
    //                 LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
    //                 IsDeletedRaw       = 0,
    //                 IsFileRaw          = 1,
    //             };
    //             itemsToInsert.Add(newItem);
    //         }
    //     }
    //
    //     if (itemsToInsert.Count == 0)
    //         return Result.Failure(DomainError.NoDataProvided());
    //
    //     var sqlBuilder = new StringBuilder();
    //     var parameters = new List<NpgsqlParameter>();
    //
    //     sqlBuilder.AppendLine(
    //         $"INSERT INTO \"ItemDSN{request.Dsn}\" (\"Id\", \"Data\", \"ForeignDsn\", \"MimeType\", \"IsForeignRaw\", \"LastChangeDateTime\", \"IsDeletedRaw\", \"IsFileRaw\") VALUES ");
    //
    //     for (var i = 0; i < itemsToInsert.Count; i++){
    //         var pId         = new NpgsqlParameter($"@Id{i}", itemsToInsert[i].Id);
    //         var pData       = new NpgsqlParameter($"@Data{i}", itemsToInsert[i].Data);
    //         var pForeignDsn = new NpgsqlParameter($"@ForeignDsn{i}", itemsToInsert[i].ForeignDsn);
    //         var pMimeType   = new NpgsqlParameter($"@MimeType{i}", itemsToInsert[i].MimeType);
    //         var pIsForeign  = new NpgsqlParameter($"@IsForeign{i}", itemsToInsert[i].IsForeignRaw);
    //         var pLastChange = new NpgsqlParameter($"@LastChangeDateTime{i}", itemsToInsert[i].LastChangeDateTime);
    //         var pIsDeleted  = new NpgsqlParameter($"@IsDeleted{i}", itemsToInsert[i].IsDeletedRaw);
    //         var pIsFile     = new NpgsqlParameter($"@IsFile{i}", itemsToInsert[i].IsFileRaw);
    //
    //         parameters.AddRange([pId, pData, pForeignDsn, pMimeType, pIsForeign, pLastChange, pIsDeleted, pIsFile,]);
    //
    //         sqlBuilder.Append(
    //             $"(@Id{i}, @Data{i}::bytea, @ForeignDsn{i}, @MimeType{i}, @IsForeign{i}, @LastChangeDateTime{i}, @IsDeleted{i}, @IsFile{i})");
    //
    //         sqlBuilder.AppendLine(i < itemsToInsert.Count - 1 ? "," : ";");
    //     }
    //
    //     await _context.Database.ExecuteSqlRawAsync(sqlBuilder.ToString(), parameters.ToArray(), cancellationToken);
    //
    //     return Result.Success(itemsToInsert.Count == 1
    //                               ? itemsToInsert[0].Id
    //                               : itemsToInsert.Select(i => i.Id).ToArray());
    // }

}