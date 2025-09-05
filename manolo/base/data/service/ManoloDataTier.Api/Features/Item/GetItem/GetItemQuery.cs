using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Item.GetItem;

public class GetItemQuery : IRequest<Result>{

    [Required]
    public required int Dsn{ get; set; }

    [Required]
    public required string Id{ get; set; }

}