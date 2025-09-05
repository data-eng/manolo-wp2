using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using ManoloDataTier.Common;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace ManoloDataTier.Storage.Model;

public class Predicate : IEntity{

#region Keys

    [MaxLength(29)]
    [JsonIgnore]
    public string Id{ get; set; }

#endregion

#region Properties

    [MaxLength(255)]
    public string Description{ get; set; }

    public required long LastChangeDateTime{ get; set; }

#endregion

    public static string GenerateId() =>
        Generator.GenerateUlidWithSuffix("PRD");

}

public class PredicateConfiguration : IEntityTypeConfiguration<Predicate>{

    public void Configure(EntityTypeBuilder<Predicate> builder){
        //Key
        builder.HasKey(x => x.Id);
    }

}