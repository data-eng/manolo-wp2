using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Alias.GetId;

public class GetIdQuery : IRequest<Result>{

    [Required]
    public required string Alias{ get; set; }

}