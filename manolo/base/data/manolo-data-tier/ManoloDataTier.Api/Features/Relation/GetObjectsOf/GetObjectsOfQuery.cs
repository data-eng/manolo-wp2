using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Relation.GetObjectsOf;

public class GetObjectsOfQuery : IRequest<Result>{

    [Required]
    public required string Subject{ get; set; }

    [Required]
    public required string Description{ get; set; }

}