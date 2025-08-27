using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Predicate.CreatePredicate;

public class CreatePredicateQuery : IRequest<Result>{

    [Required]
    public required string Description{ get; set; }

}