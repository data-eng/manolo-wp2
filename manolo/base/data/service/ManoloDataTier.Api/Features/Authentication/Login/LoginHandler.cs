using System.Security.Claims;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Api.Features.Authentication.Login;

public class LoginHandler : IRequestHandler<LoginQuery, Result>{

    private readonly ManoloDbContext      _context;
    private readonly IHttpContextAccessor _httpContextAccessor;


    public LoginHandler(ManoloDbContext context, IHttpContextAccessor httpContextAccessor){
        _context             = context;
        _httpContextAccessor = httpContextAccessor;
    }

    public async Task<Result> Handle(LoginQuery request,
                                     CancellationToken cancellationToken){

        var user = await _context.Users
                                 .AsNoTracking()
                                 .FirstOrDefaultAsync(u => u.Username == request.Username, cancellationToken);

        if (user == null || !BCrypt.Net.BCrypt.Verify(request.Password, user.PasswordHash))
            return Result.Failure(DomainError.UserDoesNotExist());

        var claims = new List<Claim>{
            new(ClaimTypes.Sid, user.Id),
            new(ClaimTypes.Name, user.Username),
            new("AccessLevel", user.AccessLevel.ToString()),
        };

        var identity = new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme);

        await _httpContextAccessor.HttpContext!.SignInAsync(
            CookieAuthenticationDefaults.AuthenticationScheme,
            new(identity)
        );

        return Result.Success();
    }

}