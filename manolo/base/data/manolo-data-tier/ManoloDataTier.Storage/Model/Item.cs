using System.Text.Json.Serialization;
using ManoloDataTier.Common;

namespace ManoloDataTier.Storage.Model;

public class Item : IEntity{

#region Key

    public required string Id{ get; set; }

#endregion

#region Properties

    public uint DataOid   { get; set; }
    public int  ForeignDsn{ get; set; }

    public string MimeType{ get; set; }

    [JsonIgnore]
    public int IsForeignRaw{ get; set; }

    public bool IsForeign => IsForeignRaw == 1;

    public required long LastChangeDateTime{ get; set; }

    [JsonIgnore]
    public int IsDeletedRaw{ get; set; }

    public bool IsDeleted => IsDeletedRaw == 1;

    [JsonIgnore]
    public int IsFileRaw{ get; set; }

    public bool IsFile => IsFileRaw == 1;

#endregion

    public static string GenerateId() =>
        Generator.GenerateUlidWithSuffix("ITS");

}