using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Relation.CreateRelation;

[Authorize(Policy = "ModeratorOrHigher")]
public class CreateRelationEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Relation")]
    [HttpPost("/createRelation")]
    public async Task<string> AsyncMethod([FromQuery] CreateRelationQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}