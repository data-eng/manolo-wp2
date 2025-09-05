using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Alias.DeleteAlias;

public class DeleteAliasQuery : IRequest<Result>{

    [Required]
    public required string Alias{ get; set; }

}