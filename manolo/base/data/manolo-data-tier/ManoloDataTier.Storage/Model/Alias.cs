using System.ComponentModel.DataAnnotations;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace ManoloDataTier.Storage.Model;

public class Alias{

#region Keys

    [MaxLength(29)]
    public required string Id{ get; set; }

#endregion

#region Properties

    [MaxLength(150)]
    public required string AliasName{ get; set; }

#endregion

}

public class AliasConfiguration : IEntityTypeConfiguration<Alias>{

    public void Configure(EntityTypeBuilder<Alias> builder){
        builder.HasKey(x => x.Id);
    }

}