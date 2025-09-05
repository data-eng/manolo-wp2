using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Predicate.GetObjectsOfPredicate;

public class GetObjectsOfPredicateQuery : IRequest<Result>{

    [Required]
    public required string Description{ get; set; }

}