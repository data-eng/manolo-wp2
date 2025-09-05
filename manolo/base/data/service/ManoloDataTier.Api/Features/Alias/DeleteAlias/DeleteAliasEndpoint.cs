using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Alias.DeleteAlias;

[Authorize]
public class DeleteAliasEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Alias")]
    [HttpDelete("/deleteAlias")]
    public async Task<string> AsyncMethod([FromQuery] DeleteAliasQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}