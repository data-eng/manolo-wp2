using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Relation.GetChildren;

public class GetChildrenQuery : IRequest<Result>{

    [Required]
    public required string Parent{ get; set; }

}