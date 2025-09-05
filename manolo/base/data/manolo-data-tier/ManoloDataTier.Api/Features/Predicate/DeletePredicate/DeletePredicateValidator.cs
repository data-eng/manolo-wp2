using FluentValidation;

namespace ManoloDataTier.Api.Features.Predicate.DeletePredicate;

public class DeletePredicateValidator : AbstractValidator<DeletePredicateQuery>{

    public DeletePredicateValidator(){
        RuleFor(m => m.Description)
            .NotEmpty()
            .NotNull()
            .MaximumLength(255);
    }

}