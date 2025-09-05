using FluentValidation;

namespace ManoloDataTier.Api.Features.Relation.GetObjectsOf;

public class GetObjectsOfValidator : AbstractValidator<GetObjectsOfQuery>{

    public GetObjectsOfValidator(){
        RuleFor(m => m.Subject)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Description)
            .NotEmpty()
            .NotNull();
    }

}