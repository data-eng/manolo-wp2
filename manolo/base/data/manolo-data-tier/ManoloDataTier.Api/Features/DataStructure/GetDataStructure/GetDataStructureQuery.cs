using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.DataStructure.GetDataStructure;

public class GetDataStructureQuery : IRequest<Result>{

    public string? Id { get; set; }
    public int     Dsn{ get; set; } = -1;

    public string? Name{ get; set; }

}