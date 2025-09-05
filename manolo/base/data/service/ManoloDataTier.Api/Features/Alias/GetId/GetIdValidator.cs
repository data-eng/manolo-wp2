using FluentValidation;

namespace ManoloDataTier.Api.Features.Alias.GetId;

public class GetIdValidator : AbstractValidator<GetIdQuery>{

    public GetIdValidator(){
        RuleFor(m => m.Alias)
            .NotEmpty()
            .NotNull();
    }

}