using FluentValidation;

namespace ManoloDataTier.Api.Features.Authentication.Login;

public class LoginValidator : AbstractValidator<LoginQuery>{

    public LoginValidator(){
        RuleFor(c => c.Username)
            .NotEmpty()
            .NotNull()
            .WithMessage("Username is required.");

        RuleFor(c => c.Password)
            .NotEmpty()
            .NotNull()
            .WithMessage("Password is required.");
    }

}