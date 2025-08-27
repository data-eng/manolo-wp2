import os

from manolo.base.utils.data_synth_utils import parser_function
from manolo.data.data_synthesis.data_synthesis import train_gan, generate_samples_class_conditioned

# read args
args, unparsed = parser_function()

# create experiments folder if needed
if args.output_path == None:
    exp_path = os.path.join(args.save_root, args.data_dir, args.experiment_name)
else:
    exp_path = os.path.join(args.output_path, args.experiment_name)

os.makedirs(exp_path, exist_ok=True)


### Train model
train_gan(args, exp_path)


### Generate class-conditioned samples
model_path = os.path.join(exp_path, "generator.pth")
generate_samples_class_conditioned(args, model_path, exp_path)