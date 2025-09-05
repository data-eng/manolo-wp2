using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Item.DeleteItem;

public class DeleteItemQuery : IRequest<Result>{

    public required int    Dsn{ get; set; }
    public required string Id { get; set; }

}