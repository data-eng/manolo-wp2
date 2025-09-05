using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Authentication.AddUser;

[ApiExplorerSettings(GroupName = "Authentication")]
public class AddUserEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Authentication")]
    [HttpPost("/addUser")]
    public async Task<string> AsyncMethod([FromQuery] AddUserQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}