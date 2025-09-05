using FluentValidation;

namespace ManoloDataTier.Api.Features.Alias.CreateUpdateAlias;

public class CreateUpdateAliasValidator : AbstractValidator<CreateUpdateAliasQuery>{

    public CreateUpdateAliasValidator(){
        RuleFor(m => m.Alias)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Id)
            .NotEmpty()
            .NotNull()
            .MaximumLength(29);
    }

}