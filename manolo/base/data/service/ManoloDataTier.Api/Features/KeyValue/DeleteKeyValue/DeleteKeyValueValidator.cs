using FluentValidation;

namespace ManoloDataTier.Api.Features.KeyValue.DeleteKeyValue;

public class DeleteKeyValueValidator : AbstractValidator<DeleteKeyValueQuery>{

    public DeleteKeyValueValidator(){
        RuleFor(m => m.Key)
            .NotEmpty()
            .NotNull();
    }

}