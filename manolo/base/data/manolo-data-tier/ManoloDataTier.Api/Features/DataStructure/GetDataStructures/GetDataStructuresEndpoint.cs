using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.DataStructure.GetDataStructures;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetDataStructuresEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "DataStructure")]
    [HttpGet("/getDataStructures")]
    public async Task<string> AsyncMethod(){
        var result = await Mediator.Send(new GetDataStructuresQuery());

        return result;
    }

}