using System.Data;
using System.Text;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using ManoloDataTier.Logic.Services;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace ManoloDataTier.Api.Features.Item.UpdateItem;

public class UpdateItemHandler : IRequestHandler<UpdateItemQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;


    public UpdateItemHandler(ManoloDbContext context, IIdResolverService idResolverService){
        _context           = context;
        _idResolverService = idResolverService;
    }

    public async Task<Result> Handle(UpdateItemQuery request, CancellationToken cancellationToken){

        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!existingDataStructure)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        var id = await _idResolverService.GetIdFromRequestAsync(request.Id, cancellationToken);

        var tableName = $"ItemDSN{request.Dsn}";

        var sqlSelect = $"""
            SELECT "DataOid" 
            FROM "{tableName}" 
            WHERE "Id" = @Id
            """;

        await using (var conn = _context.Database.GetDbConnection() as NpgsqlConnection){

            if (conn == null)
                return Result.Failure(DomainError.DatabaseConnectionError());

            if (conn.State != ConnectionState.Open)
                await conn.OpenAsync(cancellationToken);

            await using var transaction = await conn.BeginTransactionAsync(cancellationToken);

            await using var cmdSelect = new NpgsqlCommand(sqlSelect, conn, transaction);
            cmdSelect.Parameters.AddWithValue("Id", id);

            var result = await cmdSelect.ExecuteScalarAsync(cancellationToken);

            if (result == null)
                return Result.Failure(DomainError.ItemDataIsFile());

            var dataOid = Convert.ToUInt32(result);

            byte[] data;

            if (!string.IsNullOrEmpty(request.Data))
                data = Encoding.UTF8.GetBytes(request.Data);
            else if (request.DataFile is{ Length: > 0, }){
                using var ms = new MemoryStream();
                await request.DataFile.CopyToAsync(ms, cancellationToken);
                data = ms.ToArray();
            }
            else
                data =[];

            using var dataStream = new MemoryStream(data);
            await DatabaseHelpers.WriteLargeObjectAsync(conn, transaction, dataOid, dataStream, cancellationToken);

            var sqlUpdate = $"""
                UPDATE "{tableName}" 
                SET "LastChangeDateTime" = @LastChangeDateTime
                WHERE "Id" = @Id
                """;

            await using var cmdUpdate = new NpgsqlCommand(sqlUpdate, conn, transaction);
            cmdUpdate.Parameters.AddWithValue("LastChangeDateTime", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());
            cmdUpdate.Parameters.AddWithValue("Id", id);

            await cmdUpdate.ExecuteNonQueryAsync(cancellationToken);

            await transaction.CommitAsync(cancellationToken);
        }

        return Result.Success($"Item: {id} updated.");

//         var data = Array.Empty<byte>();
//
//         if (!string.IsNullOrEmpty(request.Data)){
//             data = Encoding.UTF8.GetBytes(request.Data);
//         }
//         else if (request.DataFile is{ Length: > 0, }){
//             using var memoryStream = new MemoryStream();
//             await request.DataFile.CopyToAsync(memoryStream, cancellationToken);
//             data = memoryStream.ToArray();
//         }
//
//         var tableName = $"ItemDSN{request.Dsn}";
//
//         var sql = $"""
//             UPDATE "{tableName}" 
//             SET "Data" = @Data, 
//                 "LastChangeDateTime" = @LastChangeDateTime
//             WHERE "Id" = @Id;
//             """;
//
//         var parameters = new[]{
//             new NpgsqlParameter("@Data", data), 
//             new NpgsqlParameter("@LastChangeDateTime", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()), 
//             new NpgsqlParameter("@Id", id),
//         };
//
//         await _context.Database.ExecuteSqlRawAsync(sql, parameters, cancellationToken);
//
//         return Result.Success($"Item: {id} updated.");
    }

}