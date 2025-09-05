using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.DataStructure.UpdateDataStructure;

[Authorize(Policy = "ModeratorOrHigher")]
public class UpdateDataStructureEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "DataStructure")]
    [HttpPut("/updateDataStructure")]
    public async Task<string> AsyncMethod([FromQuery] UpdateDataStructureQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}