using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.DataStructure.DeleteDataStructure;

public class DeleteDataStructureQuery : IRequest<Result>{

    public string? Id  { get; set; }
    public int     Dsn { get; set; } = -1;
    public string? Name{ get; set; }

}