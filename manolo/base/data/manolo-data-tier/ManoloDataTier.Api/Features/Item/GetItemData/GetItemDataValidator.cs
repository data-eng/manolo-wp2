using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.GetItemData;

public class GetItemDataValidator : AbstractValidator<GetItemDataQuery>{

    public GetItemDataValidator(){
        RuleFor(m => m.Id)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(0);
    }

}