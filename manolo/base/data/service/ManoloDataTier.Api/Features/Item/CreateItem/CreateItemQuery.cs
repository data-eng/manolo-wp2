using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Item.CreateItem;

public class CreateItemQuery : IRequest<Result>{

    [Required]
    public required int Dsn{ get; set; }

    public string? Data{ get; set; }

    public IFormFile? DataFile{ get; set; }

}