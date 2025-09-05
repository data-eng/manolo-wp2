using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Relation.GetSubjectsOf;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetSubjectsOfEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Predicate")]
    [HttpGet("/getSubjectsOf")]
    public async Task<string> AsyncMethod([FromQuery] GetSubjectsOfQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}