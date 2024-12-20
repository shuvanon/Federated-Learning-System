import torch
import torchvision.models as models
from timm import create_model
from torch import nn
from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights


def build_model_from_config(config):
    """
    Build the model dynamically based on the configuration.

    Args:
        config (dict): Parsed configuration data.

    Returns:
        nn.Module: PyTorch model built from the configuration.
    """
    model_type = config["model"]["type"]  # custom_cnn or transformer

    # If custom CNN is selected
    if model_type == "custom_cnn":
        params = config["model"]["parameters"]["custom_cnn"]
        model = build_model(
            model_type="custom_cnn",
            use_pre_trained_weights=False,  # Custom CNN does not use pre-trained weights
            output_dim=10,  # Number of output classes
            device="cuda",
            **params
        )
    # If Vision Transformer is selected
    elif model_type == "transformer":
        params = config["model"]["parameters"]["transformer"]
        pretrained_model = params["pretrained_model"]  # Extract the model type
        model = build_model(
            model_type=pretrained_model,  # Model type (e.g., vit_b_16)
            use_pre_trained_weights=params["use_pre_trained_weights"],
            output_dim=params["output_dim"],
            device="cuda"
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def check_input(model_name, size, batch_size=2):
    model_function = getattr(models, model_name)
    model = model_function(weights=None) if hasattr(model_function, 'weights') else model_function(pretrained=False)
    sample_input = torch.randn(batch_size, 3, size, size)

    try:
        model(sample_input)
        return True
    except:
        return False


def find_input_size(model_name, initial_size=32, max_size=1024, step=1, aggressive_step=16):
    for size in [224, 299] + list(range(initial_size, max_size, step)):
        if check_input(model_name, size):
            return size


def get_activation(activation_name: str):
    """
    Returns an activation layer based on the specified name.
    :param activation_name: Strings defining the activation type (available: ReLU, PReLU, Softmax, Sigmoid, Tanh, None)
    :return: Activation layer or None
    """
    activation_name = activation_name.lower()  # Normalize to lowercase for comparison
    if activation_name == "none" or activation_name is None:
        activation = None
    elif activation_name == "relu":
        activation = nn.ReLU()  # Correct case-sensitive activation
    elif activation_name == "prelu":
        activation = nn.PReLU()
    elif activation_name == "softmax":
        activation = nn.Softmax(dim=1)  # Specify dimension for Softmax
    elif activation_name == "sigmoid":
        activation = nn.Sigmoid()
    elif activation_name == "tanh":
        activation = nn.Tanh()
    else:
        raise ValueError(f"Activation {activation_name} not implemented")
    return activation


def get_pooling(pooling_name: str, kernel_size: int, stride: int, one_dim_input: bool) -> nn.Module:
    """
    Returns the pooling function specified by pooling_name with the specified kernel_size and stride
    :param pooling_name: Name of the pooling function
    :param kernel_size: Pixel size of the pooling filter
    :param stride: Pixel distance of movement step of the pooling filter
    :param one_dim_input: Specifies if the input is one dimensional
    :return: Pytorch pooling function
    """
    if pooling_name.lower() == "max":
        if one_dim_input:
            return nn.MaxPool1d(kernel_size, stride)
        else:
            return nn.MaxPool2d(kernel_size, stride)
    elif pooling_name.lower() == "avg":
        if one_dim_input:
            return nn.AvgPool1d(kernel_size, stride)
        else:
            return nn.AvgPool2d(kernel_size, stride)
    else:
        raise Exception("Pooling {} not implemented".format(pooling_name))


class CNN(nn.Module):
    def __init__(self, input_dim, conv, cnn_activation, pooling, layer_neurons, activations, dropout, output_dim,
                 ml_type, final_model):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()

        # Add convolutional layers
        for i in range(len(conv)):
            self.layers.append(nn.Conv2d(
                in_channels=input_dim[0] if i == 0 else conv[i - 1]["out_channels"],  # Dynamically set in_channels
                out_channels=conv[i]["out_channels"],  # Access 'out_channels' key
                kernel_size=conv[i]["kernel_size"],
                stride=conv[i]["stride"],
                padding=conv[i]["padding"]
            ))

            # Add activation function for the conv layer
            activation = get_activation(cnn_activation[i])
            if activation is not None:
                self.layers.append(activation)

            # Add pooling layer if specified
            if i < len(pooling):
                pooling_type = pooling[i]["type"]
                kernel_size = pooling[i]["kernel_size"]
                stride = pooling[i]["stride"]
                if pooling_type == "max":
                    self.layers.append(nn.MaxPool2d(kernel_size, stride))
                elif pooling_type == "avg":
                    self.layers.append(nn.AvgPool2d(kernel_size, stride))

        self.layers.append(nn.Flatten())

        # Fully connected layers
        fc_input_size = self._calculate_fc_input_size(input_dim, conv, pooling)
        for i, neurons in enumerate(layer_neurons):
            self.layers.append(nn.Linear(fc_input_size if i == 0 else layer_neurons[i - 1], neurons))
            activation = get_activation(activations[i])
            if activation is not None:
                self.layers.append(activation)
            if i < len(dropout):
                self.layers.append(nn.Dropout(dropout[i]))

        # Final output layer
        if final_model:
            self.layers.append(nn.Linear(layer_neurons[-1], output_dim))

    def _calculate_fc_input_size(self, input_dim, conv, pooling):
        """Calculate the flattened input size for the fully connected layers."""
        size = input_dim[1]  # Start with height/width of the input image
        for i in range(len(conv)):
            size = (size - conv[i]["kernel_size"] + 2 * conv[i]["padding"]) // conv[i]["stride"] + 1
            if i < len(pooling):
                size = (size - pooling[i]["kernel_size"]) // pooling[i]["stride"] + 1

        return size * size * conv[-1]["out_channels"]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(f"After {layer.__class__.__name__}: {x.shape}")
        return x


def create_vision_transformer(model_type: str, pre_trained_weights: bool, num_classes: int, device: str) -> nn.Module:
    """
    Creates a Vision Transformer (ViT) or Swin Transformer model with optional pre-trained weights.

    Args:
        model_type: Type of Vision Transformer (e.g., 'vit_b_16', 'vit_l_16', 'swin_b', 'swin_l').
        pre_trained_weights: Boolean indicating whether to use pre-trained weights.
        num_classes: Number of output classes for classification.
        device: The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        Vision Transformer or Swin Transformer model with a custom classification head.
    """
    if model_type == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pre_trained_weights else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_type == "vit_l_16":
        weights = ViT_L_16_Weights.IMAGENET1K_V1 if pre_trained_weights else None
        model = models.vit_l_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_type == "swin_b":
        model = create_model("swin_base_patch4_window7_224", weights=pre_trained_weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_type == "swin_l":
        model = create_model("swin_large_patch4_window7_224", weights=pre_trained_weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported Vision Transformer model type: {model_type}")

    return model.to(device)


def build_model(model_type: str, use_pre_trained_weights: bool, output_dim: int, device: str, **kwargs) -> nn.Module:
    """
    Builds a model based on the specified type (Custom CNN or Vision Transformer).

    Args:
        model_type: Type of model to create ('custom_cnn' or 'transformer').
        use_pre_trained_weights: Boolean indicating whether to use pre-trained weights.
        output_dim: Number of output classes.
        device: The device to run the model on.

    Returns:
        A PyTorch model of the specified type.
    """
    if model_type.lower() == "custom_cnn":
        model = CNN(**kwargs, output_dim=output_dim, ml_type="classification", final_model=True)
    elif model_type.lower() in ["vit_b_16", "vit_l_16", "swin_b", "swin_l"]:
        model = create_vision_transformer(model_type, use_pre_trained_weights, output_dim, device)
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    return model
