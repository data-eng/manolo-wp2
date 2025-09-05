using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Relation.GetRelations;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetRelationsEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Relation")]
    [HttpGet("/getRelations")]
    public async Task<string> AsyncMethod(){
        var result = await Mediator.Send(new GetRelationsQuery());

        return result;
    }

}