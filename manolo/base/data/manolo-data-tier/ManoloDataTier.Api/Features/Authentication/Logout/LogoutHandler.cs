using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;

namespace ManoloDataTier.Api.Features.Authentication.Logout;

public class LogoutHandler : IRequestHandler<LogoutQuery, Result>{

    private readonly IHttpContextAccessor _httpContextAccessor;

    public LogoutHandler(IHttpContextAccessor httpContextAccessor){
        _httpContextAccessor = httpContextAccessor;
    }

    public async Task<Result> Handle(LogoutQuery request,
                                     CancellationToken cancellationToken){

        await _httpContextAccessor.HttpContext!.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);

        return Result.Success();
    }

}