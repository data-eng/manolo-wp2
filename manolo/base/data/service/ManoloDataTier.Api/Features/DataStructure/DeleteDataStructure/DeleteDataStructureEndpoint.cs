using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.DataStructure.DeleteDataStructure;

[Authorize(Policy = "ModeratorOrHigher")]
public class DeleteDataStructureEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "DataStructure")]
    [HttpDelete("/deleteDataStructure")]
    public async Task<string> AsyncMethod([FromQuery] DeleteDataStructureQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}