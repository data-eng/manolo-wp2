using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.GetItems;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetItemsEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpGet("/getItems")]
    public async Task<string> AsyncMethod([FromQuery] GetItemsQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}