using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.DeleteItem;

public class DeleteItemValidator : AbstractValidator<DeleteItemQuery>{

    public DeleteItemValidator(){
        RuleFor(m => m.Id)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(0);
    }

}