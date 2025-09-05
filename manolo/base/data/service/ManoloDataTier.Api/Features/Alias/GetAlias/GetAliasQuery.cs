using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Alias.GetAlias;

public class GetAliasQuery : IRequest<Result>{

    [Required]
    public required string Id{ get; set; }

}