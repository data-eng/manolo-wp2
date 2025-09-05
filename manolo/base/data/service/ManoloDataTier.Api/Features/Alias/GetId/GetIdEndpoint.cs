using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Alias.GetId;

[Authorize]
public class GetIdEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Alias")]
    [HttpGet("/getId")]
    public async Task<string> AsyncMethod([FromQuery] GetIdQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}