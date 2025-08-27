using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Predicate.GetSubjectsOfPredicate;

[Authorize]
public class GetSubjectsOfPredicateEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Predicate")]
    [HttpGet("/getSubjectsOfPredicate")]
    public async Task<string> AsyncMethod([FromQuery] GetSubjectsOfPredicateQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}