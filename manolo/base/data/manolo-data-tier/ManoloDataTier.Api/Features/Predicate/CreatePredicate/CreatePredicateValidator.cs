using FluentValidation;

namespace ManoloDataTier.Api.Features.Predicate.CreatePredicate;

public class CreatePredicateValidator : AbstractValidator<CreatePredicateQuery>{

    public CreatePredicateValidator(){
        RuleFor(m => m.Description)
            .NotEmpty()
            .NotNull()
            .MaximumLength(255);
    }

}