using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.GetItemData;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetItemDataEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpGet("/getItemData")]
    public async Task<string> AsyncMethod([FromQuery] GetItemDataQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}