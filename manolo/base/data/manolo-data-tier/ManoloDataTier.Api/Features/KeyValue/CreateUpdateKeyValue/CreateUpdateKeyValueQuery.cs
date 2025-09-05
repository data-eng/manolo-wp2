using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.KeyValue.CreateUpdateKeyValue;

public class CreateUpdateKeyValueQuery : IRequest<Result>{

    [Required]
    public required string Object{ get; set; }

    [Required]
    public required string Key{ get; set; }

    [Required]
    public required string Value{ get; set; }

}