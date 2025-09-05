using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Item.GetItems;

public class GetItemsQuery : IRequest<Result>{

    [Required]
    public required int Dsn{ get; set; }

}