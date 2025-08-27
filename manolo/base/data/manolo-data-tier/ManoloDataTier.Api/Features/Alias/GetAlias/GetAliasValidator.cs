using FluentValidation;

namespace ManoloDataTier.Api.Features.Alias.GetAlias;

public class GetAliasValidator : AbstractValidator<GetAliasQuery>{

    public GetAliasValidator(){
        RuleFor(m => m.Id)
            .NotEmpty()
            .NotNull()
            .MaximumLength(29);
    }

}