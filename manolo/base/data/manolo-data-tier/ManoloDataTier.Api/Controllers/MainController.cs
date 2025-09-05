using MediatR;
using Microsoft.AspNetCore.Mvc;

namespace ManoloDataTier.Api.Controllers;

[ApiController]
[Route("[controller]")]
public abstract class MainController : ControllerBase{

    private   ISender? _mediator;
    protected ISender  Mediator => _mediator ??= HttpContext.RequestServices.GetService<ISender>()!;

}