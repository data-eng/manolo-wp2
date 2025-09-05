using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Alias.CreateUpdateAlias;

[Authorize]
public class CreateUpdateAliasEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Alias")]
    [HttpPost("/createUpdateAlias")]
    public async Task<string> AsyncMethod([FromQuery] CreateUpdateAliasQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}