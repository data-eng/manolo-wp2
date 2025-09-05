using System.Text.Json;
using System.Text.Json.Serialization;

namespace ManoloDataTier.Logic.Domains;

public class Result{

    private static readonly JsonSerializerOptions JsonSerializerOptions = new(){
        WriteIndented = true,
    };

    public Result(){ }

    protected internal Result(object? message){
        Message = message ?? "Successful Operation.";
    }

    [JsonInclude]
    public object Message{ get; protected set; } = "Successful Operation.";

    public static implicit operator string(Result result) =>
        result.Message.ToString()!;

    public static Result Success(object? message = null){
        return new(){
            Message = message switch{
                null              => "Successful Operation.",
                string strMessage => strMessage,
                _                 => JsonSerializer.Serialize(message, JsonSerializerOptions),
            },
        };
    }

    public static Result Failure(DomainError domainError) =>
        new(){
            // ReSharper disable once NullCoalescingConditionIsAlwaysNotNullAccordingToAPIContract
            Message = domainError.GetMessage() ?? "Invalid Payload.",
        };

}