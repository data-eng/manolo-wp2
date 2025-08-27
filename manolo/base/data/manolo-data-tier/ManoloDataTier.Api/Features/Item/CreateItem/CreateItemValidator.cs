using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.CreateItem;

public class CreateItemValidator : AbstractValidator<CreateItemQuery>{

    public CreateItemValidator(){
        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(0);
    }

}