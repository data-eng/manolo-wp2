using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.DownloadItemData;

[Authorize(Policy = "ModeratorOrHigher")]
public class DownloadItemDataEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Item")]
    [HttpGet("/downloadItemData")]
    public async Task<IActionResult> AsyncMethod([FromQuery] DownloadItemDataQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}