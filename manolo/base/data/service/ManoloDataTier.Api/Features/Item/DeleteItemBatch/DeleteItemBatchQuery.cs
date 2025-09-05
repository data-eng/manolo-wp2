using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Item.DeleteItemBatch;

public class DeleteItemBatchQuery : IRequest<Result>{

    public required int      Dsn{ get; set; }
    public required string[] Ids{ get; set; }

}