using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.CreateItem;

[Authorize(Policy = "ModeratorOrHigher")]
public class CreateItemEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpPost("/createItem")] 
    [RequestSizeLimit(1073741824)]
    //1GB
    public async Task<string> AsyncMethod([FromForm] CreateItemQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}