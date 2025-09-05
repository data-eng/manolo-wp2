using FluentValidation;

namespace ManoloDataTier.Api.Features.KeyValue.GetKeyValuePerObject;

public class GetKeyValuePerObjectValidator : AbstractValidator<GetKeyValuePerObjectQuery>{

    public GetKeyValuePerObjectValidator(){
        RuleFor(m => m.Obj)
            .NotEmpty()
            .NotNull();
    }

}