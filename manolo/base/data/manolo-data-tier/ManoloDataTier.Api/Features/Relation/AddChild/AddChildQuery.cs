using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Relation.AddChild;

public class AddChildQuery : IRequest<Result>{

    [Required]
    public required int Dsn{ get; set; }

    [Required]
    public required string Child{ get; set; }

    [Required]
    public required string Parent{ get; set; }

}