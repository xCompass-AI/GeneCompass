import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_degree_conv import GraphDegreeConv


class NeuralFingerprint(nn.Module):
    def __init__(self, node_size, edge_size, conv_layer_sizes, output_size, degree_list, device, batch_normalize=True):
        """
        Args:
            node_size (int): dimension of node representations
            edge_size (int): dimension of edge representations
            conv_layer_sizes (list of int): the lengths of the output vectors
                of convolutional layers
            output_size (int): length of the finger print vector
            type_map (dict string:string): type of the batch nodes, vertex nodes,
                and edge nodes
            degree_list (list of int): a list of degrees for different
                convolutional parameters
            batch_normalize (bool): enable batch normalization (default True)
        """
        super(NeuralFingerprint, self).__init__()
        self.num_layers = len(conv_layer_sizes)
        self.output_size = output_size
        self.degree_list = degree_list
        self.conv_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        self.device = device
        layers_sizes = [node_size] + conv_layer_sizes
        for input_size in layers_sizes:
            self.out_layers.append(nn.Linear(input_size, output_size))
        for prev_size, next_size in zip(layers_sizes[:-1], layers_sizes[1:]):
            self.conv_layers.append(
                GraphDegreeConv(prev_size, edge_size, next_size, degree_list, device, batch_normalize=batch_normalize))

    def forward(self, drugs):
        """
        Args:
            graph (Graph): A graph object that represents a mini-batch
        Returns:
            fingerprint: A tensor variable with shape (batch_size, output_size)
        """
        batch_size = drugs['molecules'].batch_size
        batch_idx = drugs['molecules'].get_neighbor_idx_by_batch('atom')
        molecule_length = [len(idx) for idx in batch_idx]
        max_length = max(molecule_length)
        num_atom = sum(molecule_length)
        fingerprint_atom = torch.zeros(batch_size, max_length, self.output_size).to(self.device).double()
        atom_activations = torch.zeros(num_atom, self.output_size).to(self.device).double()
        neighbor_by_degree = []
        for degree in self.degree_list:
            neighbor_by_degree.append({
                'node': drugs['molecules'].get_neighbor_idx_by_degree('atom', degree),
                'edge': drugs['molecules'].get_neighbor_idx_by_degree('bond', degree)
            })

        def fingerprint_update(linear, node_repr):
            atom_activations = F.softmax(linear(node_repr))
            return atom_activations

        node_repr = drugs['atom']
        for layer_idx in range(self.num_layers):
            # (#nodes, #output_size)
            atom_activations += fingerprint_update(self.out_layers[layer_idx], node_repr)
            node_repr = self.conv_layers[layer_idx](drugs['molecules'],  node_repr, drugs['bond'],
                                                    neighbor_by_degree)
        atom_activations += fingerprint_update(self.out_layers[-1],  node_repr)
        for idx, atom_idx in enumerate(batch_idx):
            fingerprint_atom[idx][:len(atom_idx)] = atom_activations[atom_idx, ...]
        return fingerprint_atom
