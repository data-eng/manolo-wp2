using FluentValidation;

namespace ManoloDataTier.Api.Features.Alias.DeleteAlias;

public class DeleteAliasValidator : AbstractValidator<DeleteAliasQuery>{

    public DeleteAliasValidator(){
        RuleFor(m => m.Alias)
            .NotEmpty()
            .NotNull();
    }

}