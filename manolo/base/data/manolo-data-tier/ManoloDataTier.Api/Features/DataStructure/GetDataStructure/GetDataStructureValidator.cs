using FluentValidation;

namespace ManoloDataTier.Api.Features.DataStructure.GetDataStructure;

public class GetDataStructureValidator : AbstractValidator<GetDataStructureQuery>{

    public GetDataStructureValidator(){
        RuleFor(m => m.Name)
            .MaximumLength(255);

        RuleFor(m => m.Dsn)
            .GreaterThan(-1);
    }

}