using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Interfaces;
using ManoloDataTier.Logic.Services;
using MediatR;
using Microsoft.EntityFrameworkCore;
using RelationModel = ManoloDataTier.Storage.Model.Relation;

namespace ManoloDataTier.Api.Features.Relation.CreateRelation;

public class CreateRelationHandler : IRequestHandler<CreateRelationQuery, Result>{

    private readonly ManoloDbContext    _context;
    private readonly IIdResolverService _idResolverService;
    private readonly ServiceHelpers     _serviceHelpers;


    public CreateRelationHandler(ManoloDbContext context, IIdResolverService idResolverService, ServiceHelpers serviceHelpers){
        _context           = context;
        _idResolverService = idResolverService;
        _serviceHelpers    = serviceHelpers;
    }

    public async Task<Result> Handle(CreateRelationQuery request,
                                     CancellationToken cancellationToken){

        var existingPredicate =
            await _context.Predicates
                          .Select(d => new{
                              d.Description,
                              d.Id,
                          })
                          .FirstOrDefaultAsync(d => d.Description == request.Predicate, cancellationToken);

        if (existingPredicate is null)
            return Result.Failure(DomainError.NoPredicateExist(request.Predicate));

        var dsnExists = await _context.DataStructures
                                      .AnyAsync(d => d.Dsn == request.Dsn, cancellationToken);

        if (!dsnExists)
            return Result.Failure(DomainError.DataStructureDoesNotExistDsn(request.Dsn));

        var subjId = await _idResolverService.GetIdFromRequestAsync(request.Subject, cancellationToken);
        var objId  = await _idResolverService.GetIdFromRequestAsync(request.Object, cancellationToken);

        var subjectType = subjId[^3..];
        var objectType  = objId[^3..];

        if (!await _serviceHelpers.ValidateEntity(request.Dsn, subjId, subjectType, cancellationToken))
            return Result.Failure(DomainError.SubjectDoesNotExist(subjId));

        if (!await _serviceHelpers.ValidateEntity(request.Dsn, objId, objectType, cancellationToken))
            return Result.Failure(DomainError.ObjectDoesNotExist(objId));

        var relation = new RelationModel{
            Id                 = RelationModel.GenerateId(),
            Subject            = subjId,
            Object             = objId,
            Predicate          = existingPredicate.Id,
            LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        };

        var relationExists = await _context.Relations
                                           .AnyAsync(d => d.Subject == subjId && d.Object == objId, cancellationToken);

        if (relationExists)
            return Result.Failure(DomainError.RelationAlreadyExists(subjId, objId, request.Predicate));

        _context.Relations.Add(relation);
        await _context.SaveChangesAsync(cancellationToken);

        return Result.Success();
    }

}