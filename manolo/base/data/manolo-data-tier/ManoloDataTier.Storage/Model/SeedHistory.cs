using System.ComponentModel.DataAnnotations;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace ManoloDataTier.Storage.Model;

public class SeedHistory{

#region Keys

    [MaxLength(29)]
    public required string Id{ get; set; }

#endregion

#region Properties

    public required int Version{ get; set; }

#endregion

}

public class SeedHistoryConfiguration : IEntityTypeConfiguration<SeedHistory>{

    public void Configure(EntityTypeBuilder<SeedHistory> builder){
        builder.HasKey(x => x.Id);
    }

}