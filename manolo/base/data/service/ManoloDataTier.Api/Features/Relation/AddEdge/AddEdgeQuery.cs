using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Relation.AddEdge;

public class AddEdgeQuery : IRequest<Result>{

    [Required]
    public required int Dsn{ get; set; }

    [Required]
    public required string Node1{ get; set; }

    [Required]
    public required string Node2{ get; set; }

    [Required]
    public required string Value{ get; set; }

    [Required]
    public required int IsDirected{ get; set; }

}