using FluentValidation;

namespace ManoloDataTier.Api.Features.Predicate.GetSubjectsOfPredicate;

public class GetSubjectsOfPredicateValidator : AbstractValidator<GetSubjectsOfPredicateQuery>{

    public GetSubjectsOfPredicateValidator(){
        RuleFor(m => m.Description)
            .NotEmpty()
            .NotNull()
            .MaximumLength(255);
    }

}