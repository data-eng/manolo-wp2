using ManoloDataTier.Api.Controllers;
using ManoloDataTier.Api.Features.Relation.CreateRelation;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Relation.AddChild;

[Authorize(Policy = "ModeratorOrHigher")]
public class AddChildEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Relation")]
    [HttpPost("/addChild")]
    public async Task<string> AsyncMethod([FromQuery] AddChildQuery query){
        var createRelation = new CreateRelationQuery{
            Dsn       = query.Dsn,
            Subject   = query.Child,
            Predicate = "|_",
            Object    = query.Parent,
        };

        var result = await Mediator.Send(createRelation);

        return result;
    }

}