using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.DeleteItemBatch;

public class DeleteItemBatchValidator : AbstractValidator<DeleteItemBatchQuery>{

    public DeleteItemBatchValidator(){
        RuleFor(m => m.Ids)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(0);
    }

}