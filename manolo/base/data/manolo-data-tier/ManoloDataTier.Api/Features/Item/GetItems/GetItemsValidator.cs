using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.GetItems;

public class GetItemsValidator : AbstractValidator<GetItemsQuery>{

    public GetItemsValidator(){
        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(-1);
    }

}