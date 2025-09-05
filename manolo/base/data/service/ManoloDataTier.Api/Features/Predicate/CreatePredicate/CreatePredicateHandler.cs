using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using MediatR;
using Microsoft.EntityFrameworkCore;
using PredicateModel = ManoloDataTier.Storage.Model.Predicate;

namespace ManoloDataTier.Api.Features.Predicate.CreatePredicate;

public class CreatePredicateHandler : IRequestHandler<CreatePredicateQuery, Result>{

    private readonly ManoloDbContext _context;


    public CreatePredicateHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(CreatePredicateQuery request,
                                     CancellationToken cancellationToken){

        var existingPredicate = await _context.Predicates
                                              .AnyAsync(d => d.Description == request.Description, cancellationToken);

        if (existingPredicate)
            return Result.Failure(DomainError.PredicateAlreadyExists(request.Description));

        var predicate = new PredicateModel{
            Id                 = PredicateModel.GenerateId(),
            Description        = request.Description,
            LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        };

        await _context.Predicates.AddAsync(predicate, cancellationToken);

        await _context.SaveChangesAsync(cancellationToken);

        return Result.Success(predicate.Id);
    }

}