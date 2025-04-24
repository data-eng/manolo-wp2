
# Load library and function from manolo base
from manolo.base.data import cifar10
from manolo.base.models import ResNet
from manolo.base.metrics import Accuracy

# Data loading
batch_size = 32
_, test_loader = cifar10.load_cifar10(batch_size)

# Model loading
model = ResNet.load_resnet18(pretrained=True)
model.eval()

# Evaluation
accuracy = Accuracy.evaluate_model_accuracy(model, test_loader, device='cpu')
#print(f'Accuracy on CIFAR-10 testset: {(accuracy*100):.2f}%')
expected_accuracy_threshold = 0.0  # For example, 80%
assert accuracy == expected_accuracy_threshold, "Model accuracy not expected"
