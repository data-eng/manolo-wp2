using System.Reflection;
using ManoloDataTier.Logic.Database;
using ManoloDataTier.Storage.Model;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;

namespace ManoloDataTier.Logic.Extensions;

public class DynamicTableService{

    private readonly ManoloDbContext _dbContext;


    private readonly string[] _excludedProperties =[
        "IsDeleted",
        "IsForeign",
        "IsFile",
    ];

    public DynamicTableService(ManoloDbContext dbContext){
        _dbContext = dbContext;
    }

    public void CreateDynamicTables(){
        var dataStructureIds = _dbContext.DataStructures
                                         .Where(d => d.IsDeletedRaw == 0)
                                         .Select(d => d.Dsn)
                                         .ToList();

        foreach (var createTableSql in dataStructureIds.Select(dataStructureId => $"ItemDSN{dataStructureId}")
                                                       .Select(GenerateCreateTableSql<Item>))
            _dbContext.Database.ExecuteSqlRaw(createTableSql);
    }

    private string GenerateCreateTableSql<T>(string tableName){
        var properties = typeof(T).GetProperties(BindingFlags.Public | BindingFlags.Instance);

        var sql = $"CREATE TABLE IF NOT EXISTS \"{tableName}\" (";

        foreach (var property in properties){
            if (_excludedProperties.Contains(property.Name))
                continue;

            var columnType = MapCSharpTypeToSqlType(property.PropertyType);

            sql += $"\"{property.Name}\" {columnType}, ";
        }

        sql = sql.TrimEnd(',', ' ');

        sql += ");";

        return sql;
    }

    private static string MapCSharpTypeToSqlType(Type type){
        if (type == typeof(int))
            return "INTEGER";

        if (type == typeof(string))
            return "VARCHAR(255)";

        if (type == typeof(DateTime))
            return "TIMESTAMP";

        if (type == typeof(bool))
            return "SMALLINT";

        if (type == typeof(decimal))
            return "DECIMAL";

        if (type == typeof(float))
            return "REAL";

        if (type == typeof(double))
            return "DOUBLE PRECISION";

        if (type == typeof(Dictionary<string, string>))
            return "JSONB";

        if (type == typeof(List<string>))
            return "JSONB";

        if (type == typeof(long))
            return "BIGINT";

        if (type == typeof(IFormFile))
            return "BLOB";

        if (type == typeof(byte[]))
            return "BYTEA";

        if (type == typeof(byte))
            return "SMALLINT";

        if (type == typeof(uint))
            return "BIGINT";

        return "VARCHAR(255)";
    }

}