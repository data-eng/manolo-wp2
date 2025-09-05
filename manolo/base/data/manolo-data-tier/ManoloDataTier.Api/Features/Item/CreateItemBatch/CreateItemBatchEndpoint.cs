using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.CreateItemBatch;

[Authorize(Policy = "ModeratorOrHigher")]
public class CreateItemBatchEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpPost("/createItemBatch")]
    [RequestSizeLimit(21474836480)]
    //20GB
    public async Task<string> AsyncMethod([FromBody] CreateItemBatchQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}