using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Item.RestoreItem;

public class RestoreItemQuery : IRequest<Result>{

    public required int    Dsn{ get; set; }
    public required string Id { get; set; }

}