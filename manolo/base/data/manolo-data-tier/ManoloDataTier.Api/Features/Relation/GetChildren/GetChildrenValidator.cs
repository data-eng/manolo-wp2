using FluentValidation;

namespace ManoloDataTier.Api.Features.Relation.GetChildren;

public class GetChildrenValidator : AbstractValidator<GetChildrenQuery>{

    public GetChildrenValidator(){
        RuleFor(m => m.Parent)
            .NotEmpty()
            .NotNull();
    }

}