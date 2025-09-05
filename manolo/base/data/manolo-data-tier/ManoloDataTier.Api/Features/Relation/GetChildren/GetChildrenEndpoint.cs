using ManoloDataTier.Api.Controllers;
using ManoloDataTier.Api.Features.Relation.GetSubjectsOf;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Relation.GetChildren;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetChildrenEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Relation")]
    [HttpGet("/getChildren")]
    public async Task<string> AsyncMethod([FromQuery] GetChildrenQuery query){
        var getSubjects = new GetSubjectsOfQuery{
            Object      = query.Parent,
            Description = "|_",
        };

        var result = await Mediator.Send(getSubjects);

        return result;
    }

}