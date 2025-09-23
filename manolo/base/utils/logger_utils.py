from manolo.base.wrappers.mlflow import mlflow, infer_signature
from manolo.base.wrappers.pytorch import torch
from manolo.base.wrappers.optimum import quantization_map
from manolo.base.utils.file_utils import save_checkpoint

def log_metrics(metric, parameters=False, synchronous=True):
    if parameters:
        mlflow.log_params(metric)
    elif isinstance(metric, dict):
        mlflow.log_metrics(metric, synchronous)
    elif isinstance(metric, tuple):
        mlflow.log_metric(metric[0], metric[1])
    else:
        print("Metric not saved...")


def log_model(args, epoch, model, test_top1, test_top5, best_top1, exp_path, test_loader, device):

    is_best = False
    if test_top1 > best_top1:
        best_top1 = test_top1
        is_best = True
    
    qz_map = None if args.qz_start == -1 else quantization_map(model)
    save_checkpoint({
        'epoch': epoch,
        'model': model.state_dict(),
        'prec@1': test_top1,
        'prec@5': test_top5,
    }, is_best, exp_path, qz_map)
    print(f'\nModel saved in {exp_path}\n')

    example_batch, _ = next(iter(test_loader))
    example_batch = torch.Tensor(example_batch).to(device)
    model_output = model(example_batch)
    if isinstance(model_output, tuple):
        _, model_output = model_output
        if hasattr(model, 'kd_output'):
            model.kd_output=False

    signature = infer_signature(example_batch.cpu().numpy(), model_output.cpu().detach().numpy())
    input_example = example_batch[:2].cpu().numpy()   # e.g. first two samples

    ### Still to review model naming convention
    model_name = args.exp_name
    reg_model_name = args.data_name + "_" + args.exp_name
    ###

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name=model_name,
        signature=signature,
        input_example=input_example,
        registered_model_name=reg_model_name
    )
    print(f"Model registered under run ID {mlflow.active_run().info.run_id}")


def log_data(data_src, type):
    if type=='path':
        mlflow.log_artifact(data_src)
    elif type=="numpy":
        mlflow.log(data_src)
