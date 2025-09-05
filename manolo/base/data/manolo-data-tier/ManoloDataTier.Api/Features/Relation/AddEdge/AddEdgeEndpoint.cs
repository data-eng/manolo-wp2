using ManoloDataTier.Api.Controllers;
using ManoloDataTier.Api.Features.KeyValue.CreateUpdateKeyValue;
using ManoloDataTier.Api.Features.Relation.CreateRelation;
using ManoloDataTier.Logic.Domains;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Relation.AddEdge;

[Authorize(Policy = "ModeratorOrHigher")]
public class AddEdgeEndpoint : MainController{

    [ApiExplorerSettings(GroupName = "Relation")]
    [HttpPost("/addEdge")]
    public async Task<string> AsyncMethod([FromQuery] AddEdgeQuery query){
        var createRelation = new CreateRelationQuery{
            Dsn       = query.Dsn,
            Subject   = query.Node1,
            Predicate = query.IsDirected == 0 ? "--" : "->",
            Object    = query.Node2,
        };

        await Mediator.Send(createRelation);

        var addKeyVal = new CreateUpdateKeyValueQuery{
            Object = query.Node1,
            Key    = "edge_",
            Value  = query.Value,
        };

        await Mediator.Send(addKeyVal);

        return Result.Success();
    }

}