using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Authentication.Login;

public class LoginQuery : IRequest<Result>{

    [Required]
    public required string Username{ get; set; }

    [Required]
    public required string Password{ get; set; }

}