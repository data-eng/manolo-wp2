using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.KeyValue.CreateUpdateKeyValueBatch;

[Authorize(Policy = "ModeratorOrHigher")]
public class CreateUpdateKeyValueBatchEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "KeyValue")]
    [HttpPost("/createUpdateKeyValueBatch")]
    public async Task<string> AsyncMethod([FromQuery] CreateUpdateKeyValueBatchQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}