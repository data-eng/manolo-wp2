using System.Runtime.CompilerServices;

namespace ManoloDataTier.Logic.Domains;

public class DomainError{

    private DomainError(int errorNumber, string message, [CallerMemberName] string? errorName = null){
        Code    = $"{errorNumber:D3} - {errorName}";
        Message = message;
    }

    private string Code   { get; }
    private string Message{ get; }

    public static DomainError UserAccessLevelNotAuthorized =>
        new(401, "You are not authorized to access this resource.");

    public static DomainError UserNotLoggedIn =>
        new(403, "You are not logged in.");

    internal string GetMessage() =>
        Code + " " + Message;

    public static DomainError DataStructureAlreadyExistsName(string input) =>
        new(001, $"DataStructure with name {input} already exists.");

    public static DomainError DataStructureAlreadyExistsDsn(int input) =>
        new(002, $"DataStructure with DSN {input} already exists.");

    public static DomainError DataStructureAlreadyExists(string requestName, int requestDsn) =>
        new(003, $"DataStructure with name {requestName} and DSN {requestDsn} already exists.");

    public static DomainError DataStructureDoesNotExistId(string input) =>
        new(004, $"DataStructure with ID {input} does not exist.");

    public static DomainError DataStructureDoesNotExistName(string input) =>
        new(005, $"DataStructure with Name {input} does not exist.");

    public static DomainError DataStructureDoesNotExistDsn(int input) =>
        new(006, $"DataStructure with DSN {input} does not exist.");

    public static DomainError NoDataStructures() =>
        new(007, "No DataStructures found.");

    public static DomainError ItemDataIsFile() =>
        new(100, "Item data is in a file, use the Download Option instead.");

    public static DomainError NoItemsExist(int input) =>
        new(101, $"No Items found for DSN {input}.");

    public static DomainError ItemDataIsString() =>
        new(102, "Item data is in a string format, use the Get Data Option instead.");

    public static DomainError ItemDoesNotExist(string input, int dsn) =>
        new(103, $"Item: {input} does not exist for DSN: {dsn}.");

    public static DomainError NoDataProvided() =>
        new(104, "No data provided.");

    public static DomainError PredicateAlreadyExists(string input) =>
        new(200, $"Predicate: {input} already exists.");

    public static DomainError NoPredicatesExist() =>
        new(201, "No predicates found.");

    public static DomainError NoPredicateExist(string input) =>
        new(202, $"Predicate: {input} does not exist.");

    public static DomainError AliasAlreadyExistsId(string alias) =>
        new(300, $"Alias: {alias} already exists.");

    public static DomainError AliasAlreadyExistsAlias(string id, string alias) =>
        new(301, $"Alias: {alias} already for Id: {id}.");

    public static DomainError IdDoesNotHaveAlias(string id) =>
        new(302, $"Id: {id} does not have an alias.");

    public static DomainError AliasDoesNotExist(string alias) =>
        new(303, $"Alias: {alias} does not exist.");

    public static DomainError KeyValueAlreadyExistsKey(string input) =>
        new(500, $"Key: {input} already exists.");

    public static DomainError KeyValueDoesNotExist(string input) =>
        new(501, $"Key: {input} does not exist.");

    public static DomainError ObjectNoKeys(string input) =>
        new(502, $"Object: {input} has no keys.");

    public static DomainError NoKeys() =>
        new(503, "No keys exist.");

    public static DomainError KeyValueBatchMismatch(int keyLength, int valueLength) =>
        new(504, $"Key and Value arrays must have the same length. {keyLength} != {valueLength}");

    public static DomainError KeyValueBatchDoesNotExist(string requestObject, string[] requestKeys) =>
        new(504, $"Object: {requestObject} does not have the keys: {string.Join(", ", requestKeys)}");

    public static DomainError KeyValueBatchEmpty() =>
        new(505, "Key arrays cannot be empty.");

    public static DomainError InvalidBackupData() =>
        new(600, "Invalid backup data.");

    public static DomainError UserDoesNotExist() =>
        new(700, "The credentials you provided does not match any user.");

    public static DomainError UserAlreadyExists(string input) =>
        new(701, $"User: {input} already exists.");

    public static DomainError ObjectDoesNotExist(string input) =>
        new(800, $"Object: {input} does not exist.");

    public static DomainError SubjectDoesNotExist(string input) =>
        new(801, $"Subject: {input} does not exist.");

    public static DomainError RelationDoesNotExist(string subject, string predicate, string obj) =>
        new(802, $"Relation: {subject} {predicate} {obj} does not exist.");

    public static DomainError RelationAlreadyExists(string subjId, string objjId, string requestPredicate) =>
        new(803, $"Relation: {subjId} {requestPredicate} {objjId} already exists.");

    public static DomainError NoRelations() =>
        new(804, "No relations found.");

    public static DomainError NoQueryProvided() => new(900, "No query provided.");

    public static DomainError DatabaseConnectionError() => new(999, "Database connection error.");

}