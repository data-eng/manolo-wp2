using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Relation.GetSubjectsOf;

public class GetSubjectsOfQuery : IRequest<Result>{

    [Required]
    public required string Object{ get; set; }

    [Required]
    public required string Description{ get; set; }

}