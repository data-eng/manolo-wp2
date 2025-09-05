using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.KeyValue.CreateUpdateKeyValue;

[Authorize(Policy = "ModeratorOrHigher")]
public class CreateUpdateKeyValueEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "KeyValue")]
    [HttpPost("/createUpdateKeyValue")] //TODO remove since we have batch
    public async Task<string> AsyncMethod([FromQuery] CreateUpdateKeyValueQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}