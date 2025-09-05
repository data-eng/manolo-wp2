using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.RestoreItem;

public class RestoreItemValidator : AbstractValidator<RestoreItemQuery>{

    public RestoreItemValidator(){
        RuleFor(m => m.Id)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(0);
    }

}