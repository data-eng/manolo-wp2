using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Item.UpdateItem;

public class UpdateItemQuery : IRequest<Result>{

    [Required]
    public int Dsn{ get; set; }

    public required string  Id  { get; set; }
    public          string? Data{ get; set; }

    public IFormFile? DataFile{ get; set; }

}