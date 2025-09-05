using FluentValidation;

namespace ManoloDataTier.Api.Features.Relation.GetSubjectsOf;

public class GetSubjectsOfValidator : AbstractValidator<GetSubjectsOfQuery>{

    public GetSubjectsOfValidator(){
        RuleFor(m => m.Object)
            .NotEmpty()
            .NotNull()
            .MaximumLength(29);

        RuleFor(m => m.Description)
            .NotEmpty()
            .NotNull()
            .MaximumLength(255);
    }

}