using ManoloDataTier.Api.Controllers;
using ManoloDataTier.Api.Features.Relation.GetSubjectsOf;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Relation.GetAdjacencyList;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetAdjacencyListEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Relation")]
    [HttpGet("/getAdjacencyList")]
    public async Task<string> AsyncMethod([FromQuery] GetAdjacencyListQuery query){
        var getSubjects = new GetSubjectsOfQuery{
            Object      = query.Node1,
            Description = "getAdjacencyList",
        };

        var result = await Mediator.Send(getSubjects);

        return result;
    }

}