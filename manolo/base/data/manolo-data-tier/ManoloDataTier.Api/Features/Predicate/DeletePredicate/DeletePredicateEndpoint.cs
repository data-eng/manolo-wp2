using ManoloDataTier.Api.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Predicate.DeletePredicate;

[Authorize]
public class DeletePredicateEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Predicate")]
    [HttpDelete("/deletePredicate")]
    public async Task<string> AsyncMethod([FromQuery] DeletePredicateQuery query){
        var result = await Mediator.Send(query);

        return result;
    }

}