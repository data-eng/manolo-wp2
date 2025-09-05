using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.RestoreItem;

[Authorize(Policy = "ModeratorOrHigher")]
public class RestoreItemEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpPut("/restoreItem")]
    public async Task<string> AsyncMethod([FromQuery] RestoreItemQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}