using FluentValidation;

namespace ManoloDataTier.Api.Features.Predicate.GetObjectsOfPredicate;

public class GetObjectsOfPredicateValidator : AbstractValidator<GetObjectsOfPredicateQuery>{

    public GetObjectsOfPredicateValidator(){
        RuleFor(m => m.Description)
            .NotEmpty()
            .NotNull()
            .MaximumLength(255);
    }

}