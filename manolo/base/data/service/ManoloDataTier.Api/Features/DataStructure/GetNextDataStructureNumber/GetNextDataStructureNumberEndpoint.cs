using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.DataStructure.GetNextDataStructureNumber;

[Authorize(Policy = "ModeratorOrHigher")]
public class GetNextDataStructureNumberEndpoint : MainController
{

    [ApiExplorerSettings(GroupName = "DataStructure")]
    [HttpGet("/getNextDataStructureNumber")]
    public async Task<string> AsyncMethod()
    {
        var result = await Mediator.Send(new GetNextDataStructureNumberQuery());

        return result;
    }

}