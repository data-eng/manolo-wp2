using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Predicate.GetPredicates;

[Authorize]
public class GetPredicatesEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Predicate")]
    [HttpGet("/getPredicates")]
    public async Task<string> AsyncMethod(){
        var result = await Mediator.Send(new GetPredicatesQuery());

        return result;
    }

}