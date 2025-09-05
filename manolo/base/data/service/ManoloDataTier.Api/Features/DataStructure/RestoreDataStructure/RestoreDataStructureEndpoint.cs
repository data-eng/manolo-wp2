using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.DataStructure.RestoreDataStructure;

[Authorize(Policy = "ModeratorOrHigher")]
public class RestoreDataStructureEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "DataStructure")]
    [HttpPut("/restoreDataStructure")]
    public async Task<string> AsyncMethod([FromQuery] RestoreDataStructureQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}