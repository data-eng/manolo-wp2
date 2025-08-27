using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using ManoloDataTier.Common;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace ManoloDataTier.Storage.Model;

public class User{

#region Keys

    [MaxLength(29)]
    public required string Id{ get; set; }

#endregion

#region Properties

    [MaxLength(100)]
    public required string Username{ get; set; }

    [JsonIgnore]
    [MaxLength(200)]
    public required string PasswordHash{ get; set; }

    public required byte AccessLevel{ get; set; }

#endregion


    public static string GenerateId() =>
        Generator.GenerateUlidWithSuffix("USR");

}

public class UserConfiguration : IEntityTypeConfiguration<User>{

    public void Configure(EntityTypeBuilder<User> builder){
        //Key
        builder.HasKey(x => x.Id);
    }

}