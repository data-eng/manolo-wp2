using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.DeleteItemBatch;

[Authorize(Policy = "ModeratorOrHigher")]
public class DeleteItemBatchEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpDelete("/deleteItemBatch")]
    public async Task<string> AsyncMethod([FromQuery] DeleteItemBatchQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}