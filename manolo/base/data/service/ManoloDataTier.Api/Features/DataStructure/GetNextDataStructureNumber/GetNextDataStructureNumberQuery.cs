using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.DataStructure.GetNextDataStructureNumber;

public class GetNextDataStructureNumberQuery : IRequest<Result>;