using FluentValidation;

namespace ManoloDataTier.Api.Features.Authentication.AddUser;

public class AddUserValidator : AbstractValidator<AddUserQuery>{

    public AddUserValidator(){
        RuleFor(c => c.Username)
            .NotEmpty()
            .NotNull()
            .WithMessage("Username is required.");

        RuleFor(c => c.Password)
            .NotEmpty()
            .NotNull()
            .WithMessage("Password is required.");

        RuleFor(c => c.AccessLevel)
            .NotNull()
            .WithMessage("Access level is required.");
    }

}