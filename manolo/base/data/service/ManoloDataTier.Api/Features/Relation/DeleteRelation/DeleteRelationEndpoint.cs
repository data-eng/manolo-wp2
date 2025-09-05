using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Relation.DeleteRelation;

[Authorize(Policy = "ModeratorOrHigher")]
public class DeleteRelationEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Relation")]
    [HttpDelete("/deleteRelation")]
    public async Task<string> AsyncMethod([FromQuery] DeleteRelationQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}