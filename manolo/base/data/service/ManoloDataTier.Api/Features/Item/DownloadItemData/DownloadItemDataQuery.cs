using System.ComponentModel.DataAnnotations;
using MediatR;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Features.Item.DownloadItemData;

public class DownloadItemDataQuery : IRequest<IActionResult>{

    [Required]
    public required int Dsn{ get; set; }

    [Required]
    public required string Id{ get; set; }

}