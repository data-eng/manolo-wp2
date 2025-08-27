using FluentValidation;

namespace ManoloDataTier.Api.Features.DataStructure.RestoreDataStructure;

public class RestoreDataStructureValidator : AbstractValidator<RestoreDataStructureQuery>{

    public RestoreDataStructureValidator(){
        RuleFor(m => m.Name)
            .MaximumLength(255);
    }

}