using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;
using UserModel = ManoloDataTier.Storage.Model.User;

namespace ManoloDataTier.Api.Features.Authentication.AddUser;

public class AddUserHandler : IRequestHandler<AddUserQuery, Result>{

    private readonly ManoloDbContext _context;


    public AddUserHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(AddUserQuery request,
                                     CancellationToken cancellationToken){

        var userExists = await _context.Users
                                       .AsNoTracking()
                                       .CountAsync(u => u.Username == request.Username, cancellationToken);

        if (userExists != 0)
            return Result.Failure(DomainError.UserAlreadyExists(request.Username));

        var newUser = new UserModel{
            Id           = UserModel.GenerateId(),
            Username     = request.Username,
            PasswordHash = BCrypt.Net.BCrypt.HashPassword(request.Password, 13),
            AccessLevel  = request.AccessLevel,
        };

        await _context.Users.AddAsync(newUser, cancellationToken);

        await _context.SaveChangesAsync(cancellationToken);

        return Result.Success();
    }

}