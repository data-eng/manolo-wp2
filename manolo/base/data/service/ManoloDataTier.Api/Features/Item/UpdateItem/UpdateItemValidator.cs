using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.UpdateItem;

public class UpdateItemValidator : AbstractValidator<UpdateItemQuery>{

    public UpdateItemValidator(){
        RuleFor(m => m.Id)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(-1);
    }

}