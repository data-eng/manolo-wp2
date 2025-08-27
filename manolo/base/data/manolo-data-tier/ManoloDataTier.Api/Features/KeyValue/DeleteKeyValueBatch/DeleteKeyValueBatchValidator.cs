using FluentValidation;

namespace ManoloDataTier.Api.Features.KeyValue.DeleteKeyValueBatch;

public class DeleteKeyValueBatchValidator : AbstractValidator<DeleteKeyValueBatchQuery>{

    public DeleteKeyValueBatchValidator(){
        RuleFor(m => m.Keys)
            .NotEmpty()
            .NotNull();
    }

}