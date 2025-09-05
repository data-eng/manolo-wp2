using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Predicate.CreatePredicate;

[Authorize]
public class CreatePredicateEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Predicate")]
    [HttpPost("/createPredicate")]
    public async Task<string> AsyncMethod([FromQuery] CreatePredicateQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}