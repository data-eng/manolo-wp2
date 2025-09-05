using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.DataStructure.CreateDataStructure;

[Authorize(Policy = "ModeratorOrHigher")]
public class CreateDataStructureEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "DataStructure")]
    [HttpPost("/createDataStructure")]
    public async Task<string> AsyncMethod([FromQuery] CreateDataStructureQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}