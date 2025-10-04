using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using ManoloDataTier.Common;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace ManoloDataTier.Storage.Model;

public class DataStructure : IEntity{

#region Keys

    [MaxLength(29)]
    public required string Id{ get; set; }

#endregion

#region Properties

    [MaxLength(150)]
    public required string Name{ get; set; }

    public required int Dsn{ get; set; }

    [MaxLength(150)]
    public required string Kind{ get; set; }
    
    [MaxLength(250)]
    public string? Description{ get; set; }

    public required long LastChangeDateTime{ get; set; }

    [JsonIgnore]
    public int IsDeletedRaw{ get; set; }

    public bool IsDeleted => IsDeletedRaw == 1;

#endregion

    public static string GenerateId() =>
        Generator.GenerateUlidWithSuffix("DSN");

}

public class DataStructureConfiguration : IEntityTypeConfiguration<DataStructure>{

    public void Configure(EntityTypeBuilder<DataStructure> builder){
        //Key
        builder.HasKey(x => x.Id);
    }

}