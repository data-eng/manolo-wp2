using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Predicate.GetObjectsOfPredicate;

[Authorize]
public class GetObjectsOfPredicateEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Predicate")]
    [HttpGet("/getObjectsOfPredicate")]
    public async Task<string> AsyncMethod([FromQuery] GetObjectsOfPredicateQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}