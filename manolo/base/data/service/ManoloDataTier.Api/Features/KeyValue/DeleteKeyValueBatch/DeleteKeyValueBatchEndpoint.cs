using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.KeyValue.DeleteKeyValueBatch;

[Authorize(Policy = "ModeratorOrHigher")]
public class DeleteKeyValueBatchEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "KeyValue")]
    [HttpDelete("/deleteKeyValueBatch")]
    public async Task<string> AsyncMethod([FromQuery] DeleteKeyValueBatchQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}