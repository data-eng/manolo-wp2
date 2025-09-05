using FluentValidation;

namespace ManoloDataTier.Api.Features.KeyValue.CreateUpdateKeyValue;

public class CreateUpdateKeyValueValidator : AbstractValidator<CreateUpdateKeyValueQuery>{

    public CreateUpdateKeyValueValidator(){
        RuleFor(m => m.Object)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Key)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Value)
            .NotEmpty()
            .NotNull();
    }

}