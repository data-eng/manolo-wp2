using System.ComponentModel.DataAnnotations;
using ManoloDataTier.Logic.Domains;
using MediatR;

namespace ManoloDataTier.Api.Features.DataStructure.CreateDataStructure;

public class CreateDataStructureQuery : IRequest<Result>{

    public int Dsn{ get; set; } = -1;

    [Required]
    public required string Name{ get; set; }

    [Required]
    public required string Kind{ get; set; }

}