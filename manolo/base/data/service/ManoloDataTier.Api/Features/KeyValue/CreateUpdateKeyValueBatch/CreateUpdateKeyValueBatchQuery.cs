using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.KeyValue.CreateUpdateKeyValueBatch;

public class CreateUpdateKeyValueBatchQuery : IRequest<Result>{

    [Required]
    public required string Object{ get; set; }

    [Required]
    public required string[] Keys{ get; set; }

    [Required]
    public required string[] Values{ get; set; }

}