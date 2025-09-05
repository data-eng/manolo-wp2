using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.KeyValue.GetKeys;

public class GetKeysQuery : IRequest<Result>{

    [Required]
    public required string Object{ get; set; }

}