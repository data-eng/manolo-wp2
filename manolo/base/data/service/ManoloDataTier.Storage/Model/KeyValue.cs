using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using ManoloDataTier.Common;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace ManoloDataTier.Storage.Model;

public class KeyValue : IEntity{

#region Key

    [MaxLength(29)]
    [JsonIgnore]
    public string Id{ get; set; }

#endregion

#region Properties

    [MaxLength(255)]
    public required string Object{ get; set; }

    [MaxLength(255)]
    public required string Key{ get; set; }

    [MaxLength(255)]
    public required string Value{ get; set; }

    public required long LastChangeDateTime{ get; set; }

#endregion

    public static string GenerateId() =>
        Generator.GenerateUlidWithSuffix("KVP");

}

public class KeyValueConfiguration : IEntityTypeConfiguration<KeyValue>{

    public void Configure(EntityTypeBuilder<KeyValue> builder){
        builder.HasKey(x => x.Id);
    }

}