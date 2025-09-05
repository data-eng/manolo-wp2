using System.Text.Json.Serialization;
using ItemModel = ManoloDataTier.Storage.Model.Item;

namespace ManoloDataTier.Storage.Dto;

public class ItemDto{

    public string Id{ get; set; }

    public uint DataOid{ get; set; }

    public int ForeignDsn{ get; set; }

    [JsonIgnore]
    public int IsForeignRaw{ get; set; }

    public string MimeType{ get; set; }

    public bool IsForeign => IsForeignRaw == 1;

    public long LastChangeDateTime{ get; set; }

    [JsonIgnore]
    public int IsDeletedRaw{ get; set; }

    public bool IsDeleted => IsDeletedRaw == 1;

    [JsonIgnore]
    public int IsFileRaw{ get; set; }

    public bool IsFile => IsFileRaw == 1;

    public ItemDto(){
        Id                 = string.Empty;
        ForeignDsn         = -1;
        MimeType           = string.Empty;
        IsForeignRaw       = 0;
        LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        IsDeletedRaw       = 0;
        IsFileRaw          = 0;
    }

    public ItemDto(ItemModel item){
        Id                 = item.Id;
        ForeignDsn         = item.ForeignDsn;
        MimeType           = item.MimeType;
        IsForeignRaw       = item.IsForeignRaw;
        LastChangeDateTime = item.LastChangeDateTime;
        IsDeletedRaw       = item.IsDeletedRaw;
        IsFileRaw          = item.IsFileRaw;
    }

}