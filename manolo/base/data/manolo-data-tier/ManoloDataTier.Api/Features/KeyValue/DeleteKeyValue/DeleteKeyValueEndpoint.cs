using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.KeyValue.DeleteKeyValue;

[Authorize(Policy = "ModeratorOrHigher")]
public class DeleteKeyValueEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "KeyValue")]
    [HttpDelete("/deleteKeyValue")] //TODO remove since we have batch
    public async Task<string> AsyncMethod([FromQuery] DeleteKeyValueQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}