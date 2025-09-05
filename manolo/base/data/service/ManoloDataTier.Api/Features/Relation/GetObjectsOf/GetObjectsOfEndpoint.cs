using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Relation.GetObjectsOf;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetObjectsOfEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Predicate")]
    [HttpGet("/getObjectsOf")]
    public async Task<string> AsyncMethod([FromQuery] GetObjectsOfQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}