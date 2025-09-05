using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Predicate.GetSubjectsOfPredicate;

public class GetSubjectsOfPredicateQuery : IRequest<Result>{

    [Required]
    public required string Description{ get; set; }

}