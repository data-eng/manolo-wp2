using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.KeyValue.DeleteKeyValueBatch;

public class DeleteKeyValueBatchQuery : IRequest<Result>{

    [Required]
    public required string Object{ get; set; }

    [Required]
    public required string[] Keys{ get; set; }

}