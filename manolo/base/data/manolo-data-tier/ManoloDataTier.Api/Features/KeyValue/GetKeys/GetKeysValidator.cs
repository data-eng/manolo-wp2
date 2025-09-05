using FluentValidation;

namespace ManoloDataTier.Api.Features.KeyValue.GetKeys;

public class GetKeysValidator : AbstractValidator<GetKeysQuery>{

    public GetKeysValidator(){
        RuleFor(m => m.Object)
            .NotEmpty()
            .NotNull();
    }

}