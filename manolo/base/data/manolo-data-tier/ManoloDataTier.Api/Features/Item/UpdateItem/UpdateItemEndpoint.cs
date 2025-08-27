using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.UpdateItem;

[Authorize(Policy = "ModeratorOrHigher")]
public class UpdateItemEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpPut("/updateItem")]
    public async Task<string> AsyncMethod([FromQuery] UpdateItemQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}