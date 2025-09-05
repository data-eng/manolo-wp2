using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Alias.CreateUpdateAlias;

public class CreateUpdateAliasQuery : IRequest<Result>{

    [Required]
    public required string Id{ get; set; }

    [Required]
    public required string Alias{ get; set; }

}