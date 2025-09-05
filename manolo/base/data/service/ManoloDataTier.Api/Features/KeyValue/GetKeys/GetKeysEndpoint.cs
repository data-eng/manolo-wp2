using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.KeyValue.GetKeys;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetKeysEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "KeyValue")]
    [HttpGet("/getKeys")]
    public async Task<string> AsyncMethod([FromQuery] GetKeysQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}