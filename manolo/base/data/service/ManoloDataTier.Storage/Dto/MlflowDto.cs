using System.Text.Json.Serialization;

namespace ManoloDataTier.Storage.Dto;

public class MlflowDto{

    //From top to bottom on how the response is structured from the mlflow api

#region Experiments

    public record GetExperimentResponse(
        [property: JsonPropertyName("experiment")]
        Experiment Experiment
    );


    public record Experiment(
        [property: JsonPropertyName("experiment_id")]
        string ExperimentId,
        [property: JsonPropertyName("name")]
        string Name,
        [property: JsonPropertyName("artifact_location")]
        string ArtifactLocation,
        [property: JsonPropertyName("lifecycle_stage")]
        string LifecycleStage,
        [property: JsonPropertyName("last_update_time")]
        long LastUpdateTime,
        [property: JsonPropertyName("creation_time")]
        long CreationTime,
        [property: JsonPropertyName("tags")]
        IList<ExperimentTag>? Tags
    );

    public record ExperimentTag(
        [property: JsonPropertyName("key")]
        string Key,
        [property: JsonPropertyName("value")]
        string Value
    );

#endregion

#region Runs

    public record GetRunResponse(
        [property: JsonPropertyName("run")]
        Run Run
    );

    public record Run(
        [property: JsonPropertyName("info")]
        RunInfo Info,
        [property: JsonPropertyName("data")]
        RunData Data,
        [property: JsonPropertyName("inputs")]
        RunInputs Inputs,
        [property: JsonPropertyName("outputs")]
        RunOutputs Outputs
    );

    public record RunInfo(
        [property: JsonPropertyName("run_id")]
        string RunId,
        [property: JsonPropertyName("run_name")]
        string RunName,
        [property: JsonPropertyName("experiment_id")]
        string ExperimentId,
        [property: JsonPropertyName("user_id")]
        string UserId,
        [property: JsonPropertyName("status")]
        RunStatus Status,
        [property: JsonPropertyName("start_time")]
        long StartTime,
        [property: JsonPropertyName("end_time")]
        long? EndTime,
        [property: JsonPropertyName("artifact_uri")]
        string? ArtifactUri,
        [property: JsonPropertyName("lifecycle_stage")]
        string LifecycleStage
    );

    public enum RunStatus{

        Running,
        Scheduled,
        Finished,
        Failed,
        Killed,

    }

    public record RunData(
        [property: JsonPropertyName("metrics")]
        IList<Metric>? Metrics,
        [property: JsonPropertyName("params")]
        IList<Param>? Params,
        [property: JsonPropertyName("tags")]
        IList<RunTag>? Tags
    );

    public record Metric(
        [property: JsonPropertyName("key")]
        string Key,
        [property: JsonPropertyName("value")]
        double Value,
        [property: JsonPropertyName("timestamp")]
        long Timestamp,
        [property: JsonPropertyName("step")]
        long Step
    );

    public record Param(
        [property: JsonPropertyName("key")]
        string Key,
        [property: JsonPropertyName("value")]
        string Value
    );

    public record RunTag(
        [property: JsonPropertyName("key")]
        string Key,
        [property: JsonPropertyName("value")]
        string Value
    );

    public record RunInputs(
        [property: JsonPropertyName("dataset_inputs")]
        IList<DatasetInput>? Metrics,
        [property: JsonPropertyName("model_inputs")]
        IList<ModelInput>? Params
    );

    public record DatasetInput(
        [property: JsonPropertyName("tags")]
        IList<InputTag>? Tags,
        [property: JsonPropertyName("dataset")]
        Dataset Dataset
    );

    public record InputTag(
        [property: JsonPropertyName("key")]
        string Key,
        [property: JsonPropertyName("value")]
        string Value
    );

    public record Dataset(
        [property: JsonPropertyName("name")]
        string Name,
        [property: JsonPropertyName("digest")]
        string Digest,
        [property: JsonPropertyName("source_type")]
        string SourceType,
        [property: JsonPropertyName("source")]
        string Source,
        [property: JsonPropertyName("schema")]
        string Schema,
        [property: JsonPropertyName("profile")]
        string Profile
    );

    public record ModelInput(
        [property: JsonPropertyName("model_id")]
        string? ModelId
    );

    public record RunOutputs(
        [property: JsonPropertyName("model_outputs")]
        IList<ModelOutput>? ModelOutputs
    );

    public record ModelOutput(
        [property: JsonPropertyName("model_id")]
        string? ModelId,
        [property: JsonPropertyName("step")]
        long? Step
    );

#endregion

#region RegisteredModels

    public record GetRegisteredModelResponse(
        [property: JsonPropertyName("registered_model")]
        RegisteredModel RegisteredModel
    );

    public record RegisteredModel(
        [property: JsonPropertyName("name")]
        string Name,
        [property: JsonPropertyName("creation_timestamp")]
        long CreationTimestamp,
        [property: JsonPropertyName("last_updated_timestamp")]
        long LastUpdatedTimestamp,
        [property: JsonPropertyName("user_id")]
        string UserId,
        [property: JsonPropertyName("description")]
        string? Description,
        [property: JsonPropertyName("latest_versions")]
        IList<ModelVersion>? LatestVersions,
        [property: JsonPropertyName("tags")]
        IList<RegisteredModelTag>? Tags,
        [property: JsonPropertyName("aliases")]
        IList<RegisteredModelAlias>? Aliases,
        [property: JsonPropertyName("deployment_job_id")]
        string? DeploymentJobId,
        [property: JsonPropertyName("deployment_job_state")]
        State? DeploymentJobState
    );

    public record ModelVersion(
        [property: JsonPropertyName("name")]
        string Name,
        [property: JsonPropertyName("version")]
        string Version,
        [property: JsonPropertyName("creation_timestamp")]
        long CreationTimestamp,
        [property: JsonPropertyName("last_updated_timestamp")]
        long LastUpdatedTimestamp,
        [property: JsonPropertyName("user_id")]
        string? UserId,
        [property: JsonPropertyName("current_stage")]
        string CurrentStage,
        [property: JsonPropertyName("description")]
        string? Description,
        [property: JsonPropertyName("source")]
        string? Source,
        [property: JsonPropertyName("run_id")]
        string? RunId,
        [property: JsonPropertyName("status")]
        ModelVersionStatus? Status,
        [property: JsonPropertyName("status_message")]
        string? StatusMessage,
        [property: JsonPropertyName("tags")]
        IList<ModelVersionTag>? Tags,
        [property: JsonPropertyName("run_link")]
        string? RunLink,
        [property: JsonPropertyName("aliases")]
        IList<string>? Aliases,
        [property: JsonPropertyName("model_id")]
        string? ModelId,
        [property: JsonPropertyName("model_params")]
        IList<ModelParam>? ModelParams,
        [property: JsonPropertyName("model_metrics")]
        IList<ModelMetric>? ModelMetrics,
        [property: JsonPropertyName("deployment_job_state")]
        ModelVersionDeploymentJobState? DeploymentJobState
    );

    public enum ModelVersionStatus{

        PendingRegistration,
        FailedRegistration,
        Ready,

    }

    public record ModelVersionTag(
        [property: JsonPropertyName("key")]
        string Key,
        [property: JsonPropertyName("value")]
        string Value
    );

    public record ModelParam(
        [property: JsonPropertyName("name")]
        string Name,
        [property: JsonPropertyName("value")]
        string Value
    );

    public record ModelMetric(
        [property: JsonPropertyName("key")]
        string Key,
        [property: JsonPropertyName("value")]
        string Value,
        [property: JsonPropertyName("timestamp")]
        long Timestamp,
        [property: JsonPropertyName("step")]
        long Step
    );

    public record RegisteredModelTag(
        [property: JsonPropertyName("key")]
        string Key,
        [property: JsonPropertyName("value")]
        string Value
    );

    public record ModelVersionDeploymentJobState(
        [property: JsonPropertyName("job_id")]
        string JobId,
        [property: JsonPropertyName("run_id")]
        string RunId,
        [property: JsonPropertyName("job_state")]
        State? JobState,
        [property: JsonPropertyName("run_state")]
        DeploymentRunState? RunState,
        [property: JsonPropertyName("current_task_name")]
        string? CurrentTaskName
    );

    public enum State{

        DeploymentJobConnectionStateUnspecified,
        NotSetUp,
        Connected,
        NotFound,
        RequiredParametersChanged,

    }

    public enum DeploymentRunState{

        DeploymentJobRunStateUnspecified,
        NoValidDeploymentJobFound,
        Running,
        Succeeded,
        Failed,
        Pending,
        Approval,

    }

    public record RegisteredModelAlias(
        [property: JsonPropertyName("alias")]
        string Alias,
        [property: JsonPropertyName("version")]
        string Version
    );

#endregion

#region MetricHistory

    public record GetMetricHistoryResponse(
        [property: JsonPropertyName("metrics")]
        IList<Metric>? Metrics,
        [property: JsonPropertyName("next_page_token")]
        string? NextPageToken
    );

#endregion

#region LatestModelVersions

    public record GetLatestModelVersionsResponse(
        [property: JsonPropertyName("model_versions")]
        IList<ModelVersion>? ModelVersions
    );

#endregion

}