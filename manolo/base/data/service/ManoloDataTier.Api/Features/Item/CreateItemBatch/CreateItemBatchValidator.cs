using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.CreateItemBatch;

public class CreateItemBatchValidator : AbstractValidator<CreateItemBatchQuery>{

    public CreateItemBatchValidator(){
        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(0);
    }

}