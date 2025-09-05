using FluentValidation;

namespace ManoloDataTier.Api.Features.Relation.AddEdge;

public class AddEdgeValidator : AbstractValidator<AddEdgeQuery>{

    public AddEdgeValidator(){
        RuleFor(m => m.Node1)
            .NotEmpty()
            .NotNull()
            .MaximumLength(29);

        RuleFor(m => m.Node2)
            .NotEmpty()
            .NotNull()
            .MaximumLength(29);

        RuleFor(m => m.Value)
            .NotEmpty()
            .NotNull()
            .MaximumLength(255);

        RuleFor(m => m.IsDirected)
            .NotEmpty()
            .NotNull()
            .Must(i => i is 0 or 1);
    }

}