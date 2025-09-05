using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.DataStructure.RestoreDataStructure;

public class RestoreDataStructureQuery : IRequest<Result>{

    public string? Id  { get; set; }
    public int     Dsn { get; set; } = -1;
    public string? Name{ get; set; }

}