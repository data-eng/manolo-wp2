using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Item.CreateItemBatch;

public class CreateItemBatchQuery : IRequest<Result>{

    [Required]
    public required int Dsn{ get; set; }

    public string[]? Data{ get; set; }

    public IFormFile[]? DataFiles{ get; set; }

}