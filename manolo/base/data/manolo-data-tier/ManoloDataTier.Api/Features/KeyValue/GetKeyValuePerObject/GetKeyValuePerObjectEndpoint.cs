using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.KeyValue.GetKeyValuePerObject;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetKeyValuePerObjectEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "KeyValue")]
    [HttpGet("/getKeyValuePerObject/{obj}")]
    public async Task<string> AsyncMethod(string obj){
        var query = new GetKeyValuePerObjectQuery{
            Obj = obj,
        };

        var result = await Mediator.Send(query);

        return result;
    }

}