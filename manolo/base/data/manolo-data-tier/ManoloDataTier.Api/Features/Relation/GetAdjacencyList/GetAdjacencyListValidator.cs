using FluentValidation;

namespace ManoloDataTier.Api.Features.Relation.GetAdjacencyList;

public class GetAdjacencyListValidator : AbstractValidator<GetAdjacencyListQuery>{

    public GetAdjacencyListValidator(){
        RuleFor(m => m.Node1)
            .NotEmpty()
            .NotNull();
    }

}