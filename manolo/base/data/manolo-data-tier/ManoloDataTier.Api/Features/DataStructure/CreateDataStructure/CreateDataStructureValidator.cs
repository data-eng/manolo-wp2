using FluentValidation;

namespace ManoloDataTier.Api.Features.DataStructure.CreateDataStructure;

public class CreateDataStructureValidator : AbstractValidator<CreateDataStructureQuery>{

    public CreateDataStructureValidator(){
        RuleFor(m => m.Name)
            .NotEmpty()
            .NotNull()
            .MaximumLength(255);

        RuleFor(m => m.Dsn)
            .GreaterThan(9);

        RuleFor(m => m.Kind)
            .NotEmpty()
            .NotNull()
            .MaximumLength(255);
    }

}