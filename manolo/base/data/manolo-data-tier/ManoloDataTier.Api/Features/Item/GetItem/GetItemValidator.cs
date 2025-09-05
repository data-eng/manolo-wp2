using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.GetItem;

public class GetItemValidator : AbstractValidator<GetItemQuery>{

    public GetItemValidator(){
        RuleFor(m => m.Id)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(0);
    }

}