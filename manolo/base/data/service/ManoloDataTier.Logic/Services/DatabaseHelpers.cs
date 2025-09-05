using Npgsql;
using ItemModel = ManoloDataTier.Storage.Model.Item;

namespace ManoloDataTier.Logic.Services;

public class DatabaseHelpers{

    public async static Task ProcessSingleItem(NpgsqlConnection conn, NpgsqlTransaction transaction,
                                               int dsn, Stream textStream, string mimeType,
                                               int isFile, List<string> insertedIds,
                                               CancellationToken cancellationToken){
        var item = new ItemModel{
            Id                 = ItemModel.GenerateId(),
            ForeignDsn         = -1,
            MimeType           = mimeType,
            IsForeignRaw       = 0,
            LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            IsDeletedRaw       = 0,
            IsFileRaw          = isFile,
        };

        await using var cmdCreate = new NpgsqlCommand("SELECT lo_create(0)", conn, transaction);
        var             oid       = Convert.ToUInt32(await cmdCreate.ExecuteScalarAsync(cancellationToken));

        await WriteLargeObjectAsync(conn, transaction, oid, textStream, cancellationToken);

        var sql = $"""
                INSERT INTO "ItemDSN{dsn}" 
                ("Id", "DataOid", "ForeignDsn", "MimeType", "IsForeignRaw", "LastChangeDateTime", "IsDeletedRaw", "IsFileRaw")
                VALUES (@Id, @DataOid, @ForeignDsn, @MimeType, @IsForeign, @LastChangeDateTime, @IsDeleted, @IsFile)
            """;

        await using var cmdInsert = new NpgsqlCommand(sql, conn, transaction);
        cmdInsert.Parameters.AddWithValue("Id", item.Id);
        cmdInsert.Parameters.AddWithValue("DataOid", (long)oid);
        cmdInsert.Parameters.AddWithValue("ForeignDsn", item.ForeignDsn);
        cmdInsert.Parameters.AddWithValue("MimeType", item.MimeType);
        cmdInsert.Parameters.AddWithValue("IsForeign", item.IsForeignRaw);
        cmdInsert.Parameters.AddWithValue("LastChangeDateTime", item.LastChangeDateTime);
        cmdInsert.Parameters.AddWithValue("IsDeleted", item.IsDeletedRaw);
        cmdInsert.Parameters.AddWithValue("IsFile", item.IsFileRaw);

        await cmdInsert.ExecuteNonQueryAsync(cancellationToken);

        insertedIds.Add(item.Id);
    }

    public async static Task<byte[]> ReadLargeObjectAsync(NpgsqlConnection conn, NpgsqlTransaction transaction, uint oid, CancellationToken cancellationToken){
        // (INV_READ = 262144)
        await using var cmdOpen = new NpgsqlCommand("SELECT lo_open(@oid, 262144)", conn, transaction);
        cmdOpen.Parameters.AddWithValue("oid", (int)oid);
        var fd = Convert.ToInt32(await cmdOpen.ExecuteScalarAsync(cancellationToken));

        using var ms     = new MemoryStream();
        var       buffer = new byte[81920];
        int       bytesRead;

        do{
            await using var cmdRead = new NpgsqlCommand("SELECT loread(@fd, @len)", conn, transaction);
            cmdRead.Parameters.AddWithValue("fd", fd);
            cmdRead.Parameters.AddWithValue("len", buffer.Length);
            await using var reader = await cmdRead.ExecuteReaderAsync(cancellationToken);

            if (await reader.ReadAsync(cancellationToken)){
                var chunk = reader.GetFieldValue<byte[]>(0);
                bytesRead = chunk.Length;
                ms.Write(chunk, 0, bytesRead);
            }
            else{
                bytesRead = 0;
            }
        } while (bytesRead > 0);

        return ms.ToArray();
    }

    public async static Task WriteLargeObjectAsync(
        NpgsqlConnection conn,
        NpgsqlTransaction transaction,
        uint oid,
        Stream dataStream,
        CancellationToken cancellationToken){

        // Open LO for writing (INV_WRITE = 131072)
        await using var cmdOpen = new NpgsqlCommand("SELECT lo_open(@oid, 131072)", conn, transaction);
        cmdOpen.Parameters.AddWithValue("oid", (int)oid);
        var fd = Convert.ToInt32(await cmdOpen.ExecuteScalarAsync(cancellationToken));

        var buffer = new byte[81920];
        int bytesRead;

        while (( bytesRead = await dataStream.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken) ) > 0){
            await using var cmdWrite = new NpgsqlCommand("SELECT lowrite(@fd, @data)", conn, transaction);
            cmdWrite.Parameters.AddWithValue("fd", fd);
            cmdWrite.Parameters.AddWithValue("data", buffer.AsMemory(0, bytesRead));
            await cmdWrite.ExecuteNonQueryAsync(cancellationToken);
        }

        await dataStream.DisposeAsync();
    }

}