using FluentValidation;

namespace ManoloDataTier.Api.Features.Relation.AddChild;

public class AddChildValidator : AbstractValidator<AddChildQuery>{

    public AddChildValidator(){
        RuleFor(m => m.Child)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Parent)
            .NotEmpty()
            .NotNull();
    }

}