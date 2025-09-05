using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Authentication.Logout;

[Authorize]
public class LogoutEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Authentication")]
    [HttpGet("/logout")]
    public async Task<string> AsyncMethod(){
        var result = await Mediator.Send(new LogoutQuery());

        return result;
    }

}