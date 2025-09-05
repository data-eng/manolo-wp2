using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.DataStructure.GetDataStructure;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetDataStructureEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "DataStructure")]
    [HttpGet("/getDataStructure")]
    public async Task<string> AsyncMethod([FromQuery] GetDataStructureQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}