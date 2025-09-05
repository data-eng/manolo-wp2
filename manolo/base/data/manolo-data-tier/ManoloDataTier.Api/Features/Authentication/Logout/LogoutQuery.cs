using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.Authentication.Logout;

public struct LogoutQuery : IRequest<Result>;