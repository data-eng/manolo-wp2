using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.DataStructure.UpdateDataStructure;

public class UpdateDataStructureQuery : IRequest<Result>{

    public string? Id  { get; set; }
    public string? Name{ get; set; }

    public int     Dsn { get; set; } = -1;
    public string? Kind{ get; set; }

}