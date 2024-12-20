import torch
import torchvision.models as models
from timm import create_model
from torch import nn


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
        pretrained_model = params.pop("pretrained_model")  # Extract the model type
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
    Returns an activation layer based on the specified name
    :param activation_name: Strings defining the activation type (available: relu, prelu, softmax, sigmoid, tanh, none)
    :return: Activation layer or None
    """
    if activation_name.lower() == "none" or activation_name == None:
        activation = None
    else:
        activation = getattr(nn, activation_name)()

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

    def __init__(
            self, input_dim: list, conv: list, cnn_activation: list, pooling: list, layer_neurons: list,
            activations: list,
            dropout: list, output_dim: int, ml_type, final_model: bool = False):
        """
        Initializes the CNN with the specified architecture
        :param input_dim: Dimension of the input data
        :param conv: List if lists of integers, where each sublist specifies one convolutional layer, defined by #Channels, kernel_size, stride, padding
        :param cnn_activation: List of names of activation functions for each convolutional layer
        :param pooling: List of lists of three elements. Each sublist specifies one pooling layer, defined by type, kernel_size, stride
        :param layer_neurons: List of integers specifying the number of neurons for each fully connected layer
        :param activations: List of names of activation functions for each fully connected layer
        :param dropout: List of dropout rates for each fully connected layer
        :param output_dim: Dimension of the output data
        :param final_model: Specifies if the current model should be the final model or a branch model
        """

        super(CNN, self).__init__()
        self.layers = nn.ModuleList()

        one_dim_input = len(input_dim) == 2

        for i in range(len(conv)):
            if i == 0:
                if one_dim_input:
                    self.layers.append(nn.Conv1d(input_dim[0], conv[i][0], conv[i][1], conv[i][2], conv[i][3]))
                else:
                    self.layers.append(nn.Conv2d(input_dim[0], conv[i][0], conv[i][1], conv[i][2], conv[i][3]))
            else:
                if one_dim_input:
                    self.layers.append(nn.Conv1d(conv[i - 1][0], conv[i][0], conv[i][1], conv[i][2], conv[i][3]))
                else:
                    self.layers.append(nn.Conv2d(conv[i - 1][0], conv[i][0], conv[i][1], conv[i][2], conv[i][3]))
            activation = get_activation(cnn_activation[i])
            if activation is not None:
                self.layers.append(activation)
            self.layers.append(get_pooling(pooling[i][0], pooling[i][1], pooling[i][2], one_dim_input))
        self.layers.append(nn.Flatten())

        image_size = input_dim[1]
        for i in range(len(conv)):
            image_size = int((image_size - conv[i][1] + 2 * conv[i][3]) / conv[i][2] + 1)
            image_size = int((image_size - pooling[i][1]) / pooling[i][2] + 1)

        for i in range(len(layer_neurons)):
            if i == 0:
                if one_dim_input:
                    layer = nn.Linear(image_size * conv[-1][0], layer_neurons[i])
                else:
                    layer = nn.Linear(image_size * image_size * conv[-1][0], layer_neurons[i])
            else:
                layer = nn.Linear(layer_neurons[i - 1], layer_neurons[i])

            self.layers.append(layer)

            activation = get_activation(activations[i])
            if activation != None:
                self.layers.append(activation)
            if len(dropout) > i:
                self.layers.append(nn.Dropout(dropout[i]))

        if final_model:
            if len(layer_neurons) == 0:
                if one_dim_input:
                    layer = nn.Linear(image_size * conv[-1][0], output_dim)
                else:
                    layer = nn.Linear(image_size * image_size * conv[-1][0], output_dim)
            else:
                layer = nn.Linear(layer_neurons[-1], output_dim)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network
        :param x: Input data
        :return: Output of the network
        """
        for layer in self.layers:
            x = layer(x)
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
        model = models.vit_b_16(weights=pre_trained_weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_type == "vit_l_16":
        model = models.vit_l_16(weights=pre_trained_weights)
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
