using FluentValidation;

namespace ManoloDataTier.Api.Features.DataStructure.DeleteDataStructure;

public class DeleteDataStructureValidator : AbstractValidator<DeleteDataStructureQuery>{

    public DeleteDataStructureValidator(){
        RuleFor(m => m.Name)
            .MaximumLength(255);
    }

}