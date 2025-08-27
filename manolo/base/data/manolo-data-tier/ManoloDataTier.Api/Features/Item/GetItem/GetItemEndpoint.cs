using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.GetItem;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetItemEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpGet("/getItem")]
    public async Task<string> AsyncMethod([FromQuery] GetItemQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}