using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Alias.GetAlias;

[Authorize]
public class GetAliasEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Alias")]
    [HttpGet("/getAlias")]
    public async Task<string> AsyncMethod([FromQuery] GetAliasQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}