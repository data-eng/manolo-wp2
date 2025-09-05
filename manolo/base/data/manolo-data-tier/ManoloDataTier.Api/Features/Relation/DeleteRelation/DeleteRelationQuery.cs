using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Relation.DeleteRelation;

public class DeleteRelationQuery : IRequest<Result>{

    [Required]
    public required int Dsn{ get; set; }

    [Required]
    public required string Subject{ get; set; }

    [Required]
    public required string Predicate{ get; set; }

    [Required]
    public required string Object{ get; set; }

}