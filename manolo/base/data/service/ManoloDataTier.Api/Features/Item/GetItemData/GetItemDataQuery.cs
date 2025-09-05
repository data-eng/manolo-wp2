using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Item.GetItemData;

public class GetItemDataQuery : IRequest<Result>{

    [Required]
    public required int Dsn{ get; set; }

    [Required]
    public required string Id{ get; set; }

}