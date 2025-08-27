using FluentValidation;

namespace ManoloDataTier.Api.Features.KeyValue.CreateUpdateKeyValueBatch;

public class CreateUpdateKeyValueBatchValidator : AbstractValidator<CreateUpdateKeyValueBatchQuery>{

    public CreateUpdateKeyValueBatchValidator(){
        RuleFor(m => m.Object)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Keys)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Values)
            .NotEmpty()
            .NotNull();
    }

}