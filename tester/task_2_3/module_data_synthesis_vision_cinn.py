import os

from manolo.base.utils.data_synth_utils import parser_function
from manolo.data.synth.data_synthesis import train_cinn, generate_samples_class_conditioned, generate_samples_style_conditioned

# read args
args, unparsed = parser_function()

# create experiments folder if needed
if args.output_path == None:
    exp_path = os.path.join(args.save_root, args.data_dir, args.experiment_name)
else:
    exp_path = os.path.join(args.output_path, args.experiment_name)
os.makedirs(exp_path, exist_ok=True)


### Train model
train_cinn(args, exp_path)


### Generate class-conditioned samples
model_path = os.path.join(exp_path, "{}.pt".format(args.experiment_name))
generate_samples_class_conditioned(args, model_path, exp_path)


### Generate style-conditioned samples: bold
group_name = 'bold'
class_conditioning_sample_indexes = [25, 79, 93, 131, 199, 200, 214, 222, 250, 311, 406, 437, 451, 495, 619, 580]

### Generate style-conditioned samples: italic
class_conditioning_sample_indexes = [61, 73, 79, 92, 121, 144, 167, 224, 242, 354, 355, 420, 444, 447, 464, 468]
group_name = 'italic'

generate_samples_style_conditioned(args, model_path, class_conditioning_sample_indexes, \
                                   group_name, exp_path=exp_path)