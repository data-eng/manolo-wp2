using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.KeyValue.GetKeyValuePerObject;

public class GetKeyValuePerObjectQuery : IRequest<Result>{

    [Required]
    public required string Obj{ get; set; }

}