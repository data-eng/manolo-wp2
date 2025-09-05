using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Authentication.AddUser;

public class AddUserQuery : IRequest<Result>{

    [Required]
    public required string Username{ get; set; }

    [Required]
    public required string Password{ get; set; }

    [Required]
    public required byte AccessLevel{ get; set; }

}