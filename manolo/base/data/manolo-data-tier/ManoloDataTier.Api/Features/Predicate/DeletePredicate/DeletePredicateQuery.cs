using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Predicate.DeletePredicate;

public class DeletePredicateQuery : IRequest<Result>{

    [Required]
    public required string Description{ get; set; }

}