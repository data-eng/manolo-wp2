using FluentValidation;

namespace ManoloDataTier.Api.Features.Relation.DeleteRelation;

public class DeleteRelationValidator : AbstractValidator<DeleteRelationQuery>{

    public DeleteRelationValidator(){
        RuleFor(m => m.Subject)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Predicate)
            .NotEmpty()
            .NotNull()
            .MaximumLength(255);

        RuleFor(m => m.Object)
            .NotEmpty()
            .NotNull();
    }

}