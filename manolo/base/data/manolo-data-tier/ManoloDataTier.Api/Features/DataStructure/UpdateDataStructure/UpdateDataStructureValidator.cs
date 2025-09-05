using FluentValidation;

namespace ManoloDataTier.Api.Features.DataStructure.UpdateDataStructure;

public class UpdateDataStructureValidator : AbstractValidator<UpdateDataStructureQuery>{

    public UpdateDataStructureValidator(){
        RuleFor(m => m.Name)
            .MaximumLength(255);

        RuleFor(m => m.Kind)
            .MaximumLength(255);
    }

}