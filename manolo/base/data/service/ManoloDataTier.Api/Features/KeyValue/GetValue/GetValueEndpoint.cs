using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.KeyValue.GetValue;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetValueEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "KeyValue")]
    [HttpGet("/getValue")]
    public async Task<string> AsyncMethod([FromQuery] GetValueQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}