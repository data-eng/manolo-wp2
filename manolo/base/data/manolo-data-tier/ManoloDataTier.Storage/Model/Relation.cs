using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using ManoloDataTier.Common;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace ManoloDataTier.Storage.Model;

public class Relation : IEntity{

#region Keys

    [MaxLength(29)]
    [JsonIgnore]
    public string Id{ get; set; }

#endregion

#region Properties

    [MaxLength(29)]
    public string Subject{ get; set; }

    [MaxLength(29)]
    public string Predicate{ get; set; }

    [MaxLength(255)]
    public string Object{ get; set; }

    public required long LastChangeDateTime{ get; set; }

#endregion

    public static string GenerateId() =>
        Generator.GenerateUlidWithSuffix("REL");

}

public class RelationConfiguration : IEntityTypeConfiguration<Relation>{

    public void Configure(EntityTypeBuilder<Relation> builder){
        //Key
        builder.HasKey(x => x.Id);
    }

}