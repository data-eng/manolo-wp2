using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Authentication.Login;

[ApiExplorerSettings(GroupName = "Authentication")]
public class LoginEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Authentication")]
    [HttpPost("/login")]
    public async Task<string> AsyncMethod([FromQuery] LoginQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}