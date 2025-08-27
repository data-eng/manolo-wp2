using FluentValidation;

namespace ManoloDataTier.Api.Features.Relation.CreateRelation;

public class CreateRelationValidator : AbstractValidator<CreateRelationQuery>{

    public CreateRelationValidator(){
        RuleFor(m => m.Subject)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Predicate)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Object)
            .NotEmpty()
            .NotNull();
    }

}