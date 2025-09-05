using FluentValidation;

namespace ManoloDataTier.Api.Features.Item.DownloadItemData;

public class DownloadItemDataValidator : AbstractValidator<DownloadItemDataQuery>{

    public DownloadItemDataValidator(){
        RuleFor(m => m.Id)
            .NotEmpty()
            .NotNull();

        RuleFor(m => m.Dsn)
            .NotEmpty()
            .NotNull()
            .GreaterThan(0);
    }

}