using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Relation.GetAdjacencyList;

public class GetAdjacencyListQuery : IRequest<Result>{

    [Required]
    public required string Node1{ get; set; }

}