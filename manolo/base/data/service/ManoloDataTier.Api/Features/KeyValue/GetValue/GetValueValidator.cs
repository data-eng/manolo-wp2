using FluentValidation;

namespace ManoloDataTier.Api.Features.KeyValue.GetValue;

public class GetValueValidator : AbstractValidator<GetValueQuery>{

    public GetValueValidator(){
        RuleFor(m => m.Key)
            .NotEmpty()
            .NotNull();
    }

}