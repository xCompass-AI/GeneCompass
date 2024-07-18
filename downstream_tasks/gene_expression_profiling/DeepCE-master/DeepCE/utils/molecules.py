import rdkit
from .molecule_utils import atom_features, bond_features
from collections import Iterable


degrees = [0, 1, 2, 3, 4, 5]


def node_id(smiles, idx):
    return "/".join([smiles, str(idx)])


class Node(object):
    """ Class represent graph node.
    Args:
        ntype (string): Node type
        ext_id (string): External identifier
        data: Node payload (default None)
    """

    def __init__(self, ntype, ext_id, data=None):
        self.ntype = ntype
        self.ext_id = ext_id
        self.data = data
        self.neighbors = set()

    def __str__(self):
        return ":".join([self.ext_id, self.ntype])

    def __lt__(self, other):
        return self.ntype < other.ntype or (self.ntype == other.ntype and self.ext_id < other.ext_id)

    def set_data(self, data):
        self.data = data

    def _add_neighbor(self, neighbors, new_neighbors):
        """ Add neighbor(s) for the node.
        Args:
            neighbors (Node or an iterable of Node): Old neighbor(s).
            new_neighbors (Node or an iterable of Node): Neighbor(s) to add.
            undirected (bool): If the edge is undirected (default False).
        """
        if isinstance(new_neighbors, Node):
            new_neighbors = [new_neighbors]
        if isinstance(new_neighbors, Iterable) and \
                all([isinstance(node, Node) for node in new_neighbors]):
            neighbors.update(new_neighbors)
        else:
            raise ValueError("`neighbors` has to be either a Node object \
                    or an iterable of Node objects!")

    def add_neighbors(self, new_neighbors):
        self._add_neighbor(self.neighbors, new_neighbors)

    def get_neighbors(self):
        return sorted(list(self.neighbors))

    def has_neighbor(self, node):
        return node in self.neighbors

    def clear_neighbors(self):
        self.neighbors = set()


class Molecule(object):
    def __init__(self, smiles):
        self.atom_dict = dict()
        self.bond_dict = dict()
        self.atom_list = []
        self.bond_list = []
        self.degree_nodelist = dict()
        self.read_from_smiles(smiles)

    def get_node(self, ntype, nid):
        if ntype == 'atom':
            return self.atom_dict[nid]
        elif ntype == 'bond':
            return self.bond_dict[nid]

    def has_node(self, ntype, nid):
        if ntype == 'atom':
            return nid in self.atom_dict
        elif ntype == 'bond':
            return nid in self.bond_dict

    def get_node_list(self, ntype):
        if ntype == 'atom':
            return self.atom_list
        elif ntype == 'bond':
            return self.bond_list

    def read_from_smiles(self, smiles):
        mol = rdkit.Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Could not parse SMILES string:", smiles)
        for atom in mol.GetAtoms():
            atom_node = Node('atom', node_id(smiles, atom.GetIdx()), atom_features(atom))
            self.atom_dict[node_id(smiles, atom.GetIdx())] = atom_node
            self.atom_list.append(atom_node)
        for bond in mol.GetBonds():
            src_node = self.get_node('atom', node_id(smiles, bond.GetBeginAtom().GetIdx()))
            tgt_node = self.get_node('atom', node_id(smiles, bond.GetEndAtom().GetIdx()))
            bond_node = Node('bond', node_id(smiles, bond.GetIdx()), bond_features(bond))
            bond_node.add_neighbors([src_node, tgt_node])
            src_node.add_neighbors([bond_node, tgt_node])
            tgt_node.add_neighbors([bond_node, src_node])
            self.bond_dict[node_id(smiles, bond.GetIdx())] = bond_node
            self.bond_list.append(bond_node)
        self.sort_atom_by_degree()

    def sort_atom_by_degree(self):
        node_list = self.get_node_list('atom')
        nodes_by_degree = {d: [] for d in degrees}
        for node in node_list:
            neighbor_num = len([n for n in node.get_neighbors() if n.ntype == 'atom'])
            nodes_by_degree[neighbor_num].append(node)
        sorted_nodes = []
        for degree in degrees:
            self.degree_nodelist[degree] = nodes_by_degree[degree]
            sorted_nodes.extend(nodes_by_degree[degree])
        self.atom_list = sorted_nodes


class Molecules(object):
    def __init__(self, smiles):
        self.batch_size = len(smiles)
        self.atom_dict = dict()
        self.bond_dict = dict()
        self.atom_list = []
        self.bond_list = []
        self.degree_nodelist = dict()
        self.read_from_smiles_batch(smiles)

    def add_subgraph(self, subgraph, prefix):
        """ Add a sub-graph to the current graph. """
        for ntype in ['atom', 'bond']:
            new_nodes = subgraph.get_node_list(ntype)
            for node in new_nodes:
                node.ext_id = node_id(prefix, node.ext_id)
                if ntype == 'atom':
                    self.atom_dict[node.ext_id] = node
                    self.atom_list.append(node)
                elif ntype == 'bond':
                    self.bond_dict[node.ext_id] = node
                    self.bond_list.append(node)

    def get_node_list(self, ntype):
        if ntype == 'atom':
            return self.atom_list
        elif ntype == 'bond':
            return self.bond_list

    def sort_atom_by_degree(self):
        node_list = self.get_node_list('atom')
        nodes_by_degree = {d: [] for d in degrees}
        for node in node_list:
            neighbor_num = len([n for n in node.get_neighbors() if n.ntype == 'atom'])
            nodes_by_degree[neighbor_num].append(node)
        sorted_nodes = []
        for degree in degrees:
            self.degree_nodelist[degree] = nodes_by_degree[degree]
            sorted_nodes.extend(nodes_by_degree[degree])
        self.atom_list = sorted_nodes

    def read_from_smiles_batch(self, smiles_batch):
        for idx, smiles in enumerate(smiles_batch):
            molecule = Molecule(smiles)
            self.add_subgraph(molecule, str(idx))
        self.sort_atom_by_degree()

    def get_neighbor_idx_by_degree(self, neighbor_type, degree):
        node_idx = {node.ext_id: idx for idx, node in enumerate(self.get_node_list(neighbor_type))}
        neighbor_idx = []
        for node in self.degree_nodelist[degree]:
            neighbor_idx.append([node_idx[n.ext_id] for n in node.get_neighbors() if n.ntype == neighbor_type])
        return neighbor_idx

    def get_neighbor_idx_by_batch(self, neighbor_type):
        node_idx = {node.ext_id: idx for idx, node in enumerate(self.get_node_list(neighbor_type))}
        neighbor_idx = [[] for i in range(self.batch_size)]
        for node in node_idx:
            idx = int(node.split('/')[0])
            neighbor_idx[idx].append(node_idx[node])
        return neighbor_idx
