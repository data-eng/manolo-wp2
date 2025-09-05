using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.KeyValue.GetValue;

public class GetValueQuery : IRequest<Result>{

    [Required]
    public required string Object{ get; set; }

    [Required]
    public required string Key{ get; set; }

}