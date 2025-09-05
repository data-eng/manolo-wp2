using ManoloDataTier.Logic.Database;
using ManoloDataTier.Logic.Domains;
using ManoloDataTier.Logic.Extensions;
using MediatR;
using Microsoft.EntityFrameworkCore;
using DataStructureModel = ManoloDataTier.Storage.Model.DataStructure;

namespace ManoloDataTier.Api.Features.DataStructure.CreateDataStructure;

public class CreateDataStructureHandler : IRequestHandler<CreateDataStructureQuery, Result>{

    private readonly ManoloDbContext _context;


    public CreateDataStructureHandler(ManoloDbContext context){
        _context = context;
    }

    public async Task<Result> Handle(CreateDataStructureQuery request,
                                     CancellationToken cancellationToken){

        var existingDataStructure = await _context.DataStructures
                                                  .AsNoTracking()
                                                  .FirstOrDefaultAsync(d => d.Name == request.Name || d.Dsn == request.Dsn, cancellationToken);

        switch (existingDataStructure){
            case null:
                break;

            case not null when existingDataStructure.Name == request.Name:
                return Result.Failure(DomainError.DataStructureAlreadyExistsName(request.Name));

            case not null when existingDataStructure.Dsn == request.Dsn:
                return Result.Failure(DomainError.DataStructureAlreadyExistsDsn(request.Dsn));

            default:
                return Result.Failure(DomainError.DataStructureAlreadyExists(request.Name, request.Dsn));
        }

        var dataStructure = new DataStructureModel{
            Id                 = DataStructureModel.GenerateId(),
            Name               = request.Name,
            Dsn                = request.Dsn == 0 ? await GetNextDsn() : request.Dsn,
            Kind               = request.Kind,
            LastChangeDateTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            IsDeletedRaw       = 0,
        };

        await _context.DataStructures.AddAsync(dataStructure, cancellationToken);

        await _context.SaveChangesAsync(cancellationToken);

        //Make the Item's Table
        var dynamicTableService = new DynamicTableService(_context);
        dynamicTableService.CreateDynamicTables();

        return Result.Success(dataStructure.Id);
    }

    private async Task<int> GetNextDsn(){
        var maxDsn = await _context.DataStructures
                                   .AsNoTracking()
                                   .Select(d => d.Dsn)
                                   .DefaultIfEmpty(9)
                                   .MaxAsync();

        return maxDsn + 1;
    }

}