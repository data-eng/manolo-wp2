using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.DeleteItem;

[Authorize(Policy = "ModeratorOrHigher")]
public class DeleteItemEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpDelete("/deleteItem")] //TODO remove since we have batch
    public async Task<string> AsyncMethod([FromQuery] DeleteItemQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}