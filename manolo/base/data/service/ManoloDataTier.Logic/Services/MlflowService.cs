using System.Text.Json;
using System.Text.Json.Serialization;
using ManoloDataTier.Logic.Settings;
using ManoloDataTier.Storage.Dto;
using Microsoft.Extensions.Options;

namespace ManoloDataTier.Logic.Services;

public class MlflowService{

    private readonly HttpClient             _httpClient;
    private readonly string                 _mlflowBaseUrl;
    private readonly JsonSerializerOptions? _jsonOptions;

    public MlflowService(HttpClient httpClient, IOptions<MlflowSettings> settings){
        _httpClient    = httpClient;
        _mlflowBaseUrl = settings.Value.MlflowUrl.TrimEnd('/') ?? "";

        _jsonOptions = new(){
            PropertyNameCaseInsensitive = true,
            Converters ={
                new JsonStringEnumConverter(JsonNamingPolicy.CamelCase, false),
            },
        };
    }

    public async Task<string> GetDataByEntityAsync(string? entity){
        var parts = entity?.Replace("mlflow://", "").Split('/');

        if (parts == null || parts.Length < 2){
            return string.Empty;
        }

        var type = parts[0];

        switch (type.ToLowerInvariant()){
            case "experiment":
                if (parts.Length != 2) return string.Empty;

                var experiment = await GetExperimentAsync(parts[1]);

                return JsonSerializer.Serialize(experiment, _jsonOptions);

            case "model":
                if (parts.Length != 2) return string.Empty;

                var model = await GetRegisteredModelAsync(parts[1]);

                return JsonSerializer.Serialize(model, _jsonOptions);

            case "run":
                if (parts.Length != 2) return string.Empty;

                var run = await GetRunAsync(parts[1]);

                return JsonSerializer.Serialize(run, _jsonOptions);

            case "metric":
                if (parts.Length < 3) return string.Empty;

                var runId     = parts[1];
                var metricKey = parts[2];
                var pageToken = parts.Length > 3 ? parts[3] : string.Empty;

                var metrics = await GetMetricAsync(runId, metricKey, pageToken);

                return JsonSerializer.Serialize(metrics, _jsonOptions);

            case "latest_model_version":
                if (parts.Length != 2) return string.Empty;

                var name = parts[1];

                var latestModelVersion = await GetLatestModelVersionAsync(name);

                return JsonSerializer.Serialize(latestModelVersion, _jsonOptions);

            default:
                return string.Empty;
        }
    }

    private async Task<MlflowDto.GetLatestModelVersionsResponse?> GetLatestModelVersionAsync(string name){
        var url = $"{_mlflowBaseUrl}/api/2.0/mlflow/registered-models/get-latest-versions?name={name}";

        var response = await _httpClient.GetAsync(url);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();

        var runsResponse = JsonSerializer.Deserialize<MlflowDto.GetLatestModelVersionsResponse>(json, _jsonOptions);

        return runsResponse;
    }


    private async Task<MlflowDto.GetMetricHistoryResponse?> GetMetricAsync(string runId, string metricKey,
                                                                           string pageToken,
                                                                           int maxResults = 10){
        var url =
            $"{_mlflowBaseUrl}/api/2.0/mlflow/runs/get-metrics?run_id={runId}&key={metricKey}&page_token={pageToken}&max_results={maxResults}";

        var response = await _httpClient.GetAsync(url);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();

        var runsResponse = JsonSerializer.Deserialize<MlflowDto.GetMetricHistoryResponse>(json, _jsonOptions);

        return runsResponse;
    }

    private async Task<MlflowDto.GetExperimentResponse?> GetExperimentAsync(string experimentId){
        var url      = $"{_mlflowBaseUrl}/api/2.0/mlflow/experiments/get?experiment_id={experimentId}";
        var response = await _httpClient.GetAsync(url);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();

        var experimentsResponse = JsonSerializer.Deserialize<MlflowDto.GetExperimentResponse>(json, _jsonOptions);

        return experimentsResponse;
    }

    private async Task<MlflowDto.GetRegisteredModelResponse?> GetRegisteredModelAsync(string modelName){
        var url = $"{_mlflowBaseUrl}/api/2.0/mlflow/registered-models/get?name={modelName}";

        var response = await _httpClient.GetAsync(url);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();

        var runsResponse = JsonSerializer.Deserialize<MlflowDto.GetRegisteredModelResponse>(json, _jsonOptions);

        return runsResponse;
    }

    private async Task<MlflowDto.GetRunResponse?> GetRunAsync(string runId){
        var url = $"{_mlflowBaseUrl}/api/2.0/mlflow/runs/get?run_id={runId}";

        var response = await _httpClient.GetAsync(url);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();

        var runResponse = JsonSerializer.Deserialize<MlflowDto.GetRunResponse>(json, _jsonOptions);

        return runResponse;
    }

}