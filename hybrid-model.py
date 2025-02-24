import torch
import torch.nn as nn
import pennylane as qml
from torch.nn import functional as F

# Define number of qubits and the quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# Define a quantum node (qnode) that uses AngleEmbedding and BasicEntanglerLayers
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


n_qubits_amp = 2


@qml.qnode(dev)
def qnode_ampl(inputs, weights):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits_amp), normalize=True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits_amp))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits_amp)]

    # Define number of layers n_layers=6


n_layers_10 = 10
weight_shapes_q = {"weights": (n_layers_10, n_qubits)}
weight_shapes_amp = {"weights": (n_layers_10, n_qubits_amp, 3)}

# Define a quantum layer (qlayer) that uses the qnode
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes_q)
q_layer_ampl = qml.qnn.TorchLayer(qnode_ampl, weight_shapes_amp)


class HybridLearnerBig(nn.Module):
    def __init__(self, qlayer=qlayer, q_layer_ampl=q_layer_ampl, input_dim=2048, drop_p=0.0, output_dim=4):

        super(HybridLearnerBig, self).__init__()

        self.drop_p = drop_p

        # Pre-net: a classical network that processes the input before the quantum layer
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )

        # Quantum layer: a quantum network (hybrid)
        self.qlayer = qlayer
        # self.reshape = nn.Linear(output_dim, 2)
        self.q_layer_ampl = q_layer_ampl

        # Post-net: a classical network that processes the output of the quantum layer
        self.post_net = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Initialize the weights
        self.weight_init()

    def weight_init(self):
        """Initialize the weights of the pre-net and post-net using Xavier initialization."""
        for layer in list(self.pre_net) + list(self.post_net):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        """Forward pass through the network."""

        x = self.pre_net(x)

        # Quantum layer operation
        x = self.qlayer(x)
        # x = self.reshape(x)
        x = self.q_layer_ampl(x)

        x = self.post_net(x)

        return torch.sigmoid(x)





def incase_I_mess_up():
    # import torch
    # import torch.nn as nn
    # import pennylane as qml
    # from torch.nn import functional as F
    #
    # # Define number of qubits and the quantum device
    # n_qubits = 4
    # dev = qml.device("default.qubit", wires=n_qubits)
    #
    # # Define a quantum node (qnode) that uses AngleEmbedding and BasicEntanglerLayers
    # @qml.qnode(dev)
    # def qnode(inputs, weights):
    #     qml.AngleEmbedding(inputs, wires=range(n_qubits))
    #     qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    #     return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    #
    # n_qubits_amp = 2
    #
    # @qml.qnode(dev)
    # def qnode_ampl(inputs, weights):
    #     qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits_amp), normalize=True)
    #     qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits_amp))
    #     return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits_amp)]
    #
    #     # Define number of layers n_layers=6
    #
    # n_layers_10 = 10
    # weight_shapes_q = {"weights": (n_layers_10, n_qubits)}
    # weight_shapes_amp = {"weights": (n_layers_10, n_qubits_amp, 3)}
    #
    # # Define a quantum layer (qlayer) that uses the qnode
    # qlayer = qml.qnn.TorchLayer(qnode, weight_shapes_q)
    # q_layer_ampl = qml.qnn.TorchLayer(qnode_ampl, weight_shapes_amp)

    # class HybridLearnerBig(nn.Module):
    #     def __init__(self, qlayer=qlayer, q_layer_ampl=q_layer_ampl, input_dim=2048, drop_p=0.0, output_dim=4):
    #
    #         super(HybridLearnerBig, self).__init__()
    #
    #         self.drop_p = drop_p
    #
    #         # Pre-net: a classical network that processes the input before the quantum layer
    #         self.pre_net = nn.Sequential(
    #             nn.Linear(input_dim, 512),
    #             nn.ReLU(),
    #             nn.Dropout(drop_p),
    #             nn.Linear(512, 32),
    #             nn.ReLU(),
    #             nn.Linear(32, output_dim),
    #             nn.ReLU()
    #         )
    #
    #         # Quantum layer: a quantum network (hybrid)
    #         self.qlayer = qlayer
    #         # self.reshape = nn.Linear(output_dim, 2)
    #         self.q_layer_ampl = q_layer_ampl
    #
    #         # Post-net: a classical network that processes the output of the quantum layer
    #         self.post_net = nn.Sequential(
    #             nn.ReLU(),
    #             nn.Dropout(drop_p),
    #             nn.Linear(2, 16),
    #             nn.Linear(16, 1),
    #             nn.Sigmoid()
    #         )
    #
    #         # Initialize the weights
    #         self.weight_init()
    #
    #         # Register the parameters
    #         self.vars = nn.ParameterList()
    #         for param in list(self.pre_net.parameters()) + list(self.post_net.parameters()):
    #             self.vars.append(param)
    #
    #     def weight_init(self):
    #         """Initialize the weights of the pre-net and post-net using Xavier initialization."""
    #         for layer in list(self.pre_net) + list(self.post_net):
    #             if isinstance(layer, nn.Linear):
    #                 nn.init.xavier_normal_(layer.weight)
    #
    #     def forward(self, x, vars=None):
    #         """Forward pass through the network."""
    #         if vars is None:
    #             vars = self.vars
    #
    #         # Pre-net operations
    #         x = F.relu(F.linear(x, vars[0], vars[1]))
    #         x = F.dropout(x, self.drop_p, training=self.training)
    #         x = F.relu(F.linear(x, vars[2], vars[3]))
    #         x = F.dropout(x, (self.drop_p - 0.2), training=self.training)
    #         x = F.relu(F.linear(x, vars[4], vars[5]))
    #
    #         x = pre_net()
    #
    #         # Quantum layer operation
    #         x = self.qlayer(x)
    #         # x = self.reshape(x)
    #         x = self.q_layer_ampl(x)
    #
    #         # Post-net operations
    #         # x = F.relu(x)
    #         x = F.dropout(x, self.drop_p, training=self.training)
    #         x = F.relu(F.linear(x, vars[6], vars[7]))
    #         x = F.relu(F.linear(x, vars[8], vars[9]))
    #
    #         return torch.sigmoid(x)
    #
    #     def parameters(self):
    #         """Override the initial parameters method to return parameters as a ParameterList."""
    #         return self.vars
    pass

