__all__ = [
    "DataJoinGraph"
]

#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import List, Dict, Set, Union, Optional
from graphviz import Digraph

from ..utils.autoai.errors import OBMMainNodeAlreadySet
from ..utils.autoai.enums import TShirtSize, PredictionType
from ..utils.autoai.utils import is_ipython, check_graphviz_binaries


class BaseOBMJson:
    """Base class for helper objects representation."""

    def __repr__(self):
        return self.__str__()


class Node(BaseOBMJson):
    """Node class for json representation and conversion of graphviz nodes."""

    def __init__(self, name: str, timestamp_column_name: str = None, timestamp_format: str = None,
                 csv_separator: str = None, sheet_name: str = None):
        self.table = Table(name=name, timestamp_column_name=timestamp_column_name, timestamp_format=timestamp_format,
                           csv_separator=csv_separator, sheet_name=sheet_name)

    def to_dict(self):
        """Convert this Node to dictionary for further REST API call."""
        return {"table_name": self.table.name}

    def _pretty_print(self, main: bool = False) -> str:
        if main:
            return f"data_join_graph.node(name='{self.table.name}', main=True)\n"

        else:
            return f"data_join_graph.node(name='{self.table.name}')\n"

    def __str__(self):
        return (f"\n\t\tNODE:\n"
                f"\t\t\t{self.table.__str__()}")


class Edge(BaseOBMJson):
    """Edge helper class for json representation of graphviz edge."""

    def __init__(self, from_node: str, to_node: str, from_column: List[str], to_column: List[str]):
        if not isinstance(from_column, list) or not isinstance(to_column, list):
            raise TypeError("\"from_column\" and \"to_column\" need to be of type List[str].")

        self.from_node = from_node
        self.to_node = to_node
        self.from_column = from_column
        self.to_column = to_column

    def to_dict(self):
        """Convert this Node to dictionary for further REST API call."""
        _dict = {
            "from": self.from_node,
            "to": self.to_node,
            "from_column": self.from_column,
            "to_column": self.to_column,
        }
        return _dict

    def _pretty_print(self) -> str:
        return f"data_join_graph.edge(from_node='{self.from_node}', to_node='{self.to_node}', from_column={self.from_column}, to_column={self.to_column})\n"

    def __str__(self):
        return (f"\n\t\tEDGE:\n"
                f"\t\t\tFROM: {self.from_node}\n"
                f"\t\t\tTO: {self.to_node}\n"
                f"\t\t\tFROM COLUMN: {self.from_column}\n"
                f"\t\t\tTO COLUMN: {self.to_column}")


class Table(BaseOBMJson):
    """Table class to represent / define OBM tables."""

    def __init__(self, name: str, timestamp_column_name: str = None, timestamp_format: str = None,
                 csv_separator: str = None, sheet_name: str = None):
        if (timestamp_column_name or timestamp_format) and not (timestamp_column_name and timestamp_format):
            print("Need to pass both column name and date format in order to mark column as timestamp type.")

        self.name = name
        self.timestamp_column_name = timestamp_column_name
        self.timestamp_format = timestamp_format
        self.csv_separator = csv_separator
        self.sheet_name = sheet_name
        self.source = {}

    def to_dict(self):
        """Convert this Node to dictionary for further REST API call."""
        _dict = {
            "table_source": self.source
        }
        if self.timestamp_column_name and self.timestamp_format:
            _dict.update({
                "column_format": {
                    self.timestamp_column_name: self.timestamp_format,
                },
                "timestamp_column_name": self.timestamp_column_name
            })
        if self.csv_separator:
            _dict.update({
                "field_delimiter": self.csv_separator
            })

        if self.sheet_name:
            _dict.update({
                "sheet_name": self.sheet_name
            })

        return _dict

    def __str__(self):
        return (f"TABLE:\n"
                f"\t\t\t\tNAME: {self.name}\n"
                f"\t\t\t\tSOURCE: {self.source}\n" +
                (
                        f"\t\t\t\tTIMESTAMP_COLUMN_NAME: \'{self.timestamp_column_name}\'\n"
                        f"\t\t\t\tCOLUMN_FORMAT: {{\n\t\t\t\t\t'{self.timestamp_column_name}': '{self.timestamp_format}'\n"
                        f"\t\t\t\t}}" if self.timestamp_column_name and self.timestamp_format else ""
                ) +
                (f"\t\t\t\tFIELD_DELIMITER: '{self.csv_separator}'\n" if self.csv_separator else "") +
                (f"\t\t\t\tSHEET_NAME: '{self.sheet_name}'\n" if self.sheet_name else "")
        )


class DataJoinGraph(Digraph, BaseOBMJson):
    """DataJoinGraph class - helper class for handling multiple data sources for AutoAI experiment.

    You can define the overall relations between each of data source and see these defined relations
    in a form of string representation calling print(ObmGraph) or to leverage graphviz library
    and make entire graph visualization.

    :param t_shirt_size: the size of the computation POD
    :type t_shirt_size: enum TShirtSize, optional


    **Example**

    .. code-block:: python

        data_join_graph = DataJoinGraph()
        # or
        data_join_graph = DataJoinGraph(t_shirt_size=DataJoinGraph.TShirtSize.L)

        data_join_graph.node(name="main")
        data_join_graph.node(name="customers")
        data_join_graph.node(name="transactions")
        data_join_graph.node(name="purchases")
        data_join_graph.node(name="products")

        data_join_graph.edge(from_node="main", to_node="customers",
                             from_column=["group_customer_id"], to_column=["group_customer_id"])
        data_join_graph.edge(from_node="main", to_node="transactions",
                             from_column=["transaction_id"], to_column=["transaction_id"])
        data_join_graph.edge(from_node="main", to_node="purchases",
                             from_column=["group_id"], to_column=["group_id"])
        data_join_graph.edge(from_node="transactions", to_node="products",
                             from_column=["product_id"], to_column=["product_id"])

        print(data_join_graph)
        data_join_graph.visualize()
    """
    TShirtSize = TShirtSize
    NodeTemplate = """<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="1">
      <TR>
          <TD STYLE="ROUNDED" BGCOLOR=\"{color}\" COLSPAN=\"{colspan}\">{dataset_name}</TD>
      </TR>
      <TR>
      </TR>
    </TABLE>>"""
    ColumnsIndex = NodeTemplate.rfind("<TR>") + len("<TR>")
    ColumnTemplate = "<TD STYLE=\"ROUNDED\" PORT=\"{port}\">{column_name}</TD>"

    def __init__(self,
                 t_shirt_size: 'TShirtSize' = TShirtSize.M,
                 max_depth: int = 3,
                 data_source_type: str = "csv",
                 worker_nodes_number: int = 1):
        super().__init__(comment='Data Join Graph')

        self.t_shirt_size = t_shirt_size
        self.worker_nodes_number = worker_nodes_number

        # note: these values should be provided by optimizer
        self.target_column = None
        self.__problem_type = None
        # --- end note

        self.main_node_name = None  # main node by default the first one
        self.main_set = False
        self.max_depth = max_depth
        self.data_source_type = data_source_type

        self.nodes: List['Node'] = []
        self.edges: List['Edge'] = []
        self.edges_to_ports: Dict[str, str] = {}
        self.node_edges: Dict[str, Set[str]] = {}

        self.fairness_info = None

    @property
    def problem_type(self) -> 'PredictionType':
        return self.__problem_type

    @problem_type.setter
    def problem_type(self, value: 'PredictionType') -> None:
        """We need to map prediction types between KB and OBM."""
        if value == PredictionType.MULTICLASS:
            self.__problem_type = 'classification'
        else:
            self.__problem_type = value

    def node(self,
             name: str,
             timestamp_column_name: Optional[str] = None,
             timestamp_format: Optional[str] = None,
             csv_separator: Optional[str] = None,
             sheet_name: Optional[str] = None,
             main: Optional[bool] = False) -> None:
        """Add node to the graph. The node is representing the particular data source for training.
        The node added as a first one will be set as main one. If you want to set another node as a main, please
        set a `main` argument to True.

        :param name: the name/id of the node, it must be the same as id passed to particular linked DataConnection
        :type name: str
        :param timestamp_column_name: the name of the column, which includes dates
        :type timestamp_column_name: str, optional
        :param timestamp_format: format of dates contained in column named 'timestamp_column_name'
        :type timestamp_format: str, optional
        :param csv_separator: separator used in source csv file
        :type csv_separator: str, optional
        :param sheet_name: used for xlsx files to indicate which sheet is used
        :type sheet_name: str, optional
        :param main: if you want to set this node as a main one, please set a `main` argument to `True`
        :type main: bool, optional

        **Example**

        .. code-block:: python

            data_join_graph.node(name="main")
        """
        self.node_edges[name] = set()

        # note: if we do not have any nodes already added, first one will be the main node
        if not self.nodes:
            self.main_node_name = name

        # override the default main node
        if main and not self.main_set:
            self.main_node_name = name
            self.main_set = True

        elif main:
            raise OBMMainNodeAlreadySet(self.main_node_name)

        self.nodes.append(Node(name=name, timestamp_column_name=timestamp_column_name,
                               timestamp_format=timestamp_format, csv_separator=csv_separator, sheet_name=sheet_name))
        # --- end note

    def edge(self, from_node, to_node, from_column: List[str], to_column: List[str]):
        """Add edge to the graph. The edge defines the connection between two DataConnections.
        eg. main --- from column customer_id to column customer_id --> customers

        :param from_node: the starting Node
        :type from_node: str

        :param to_node: the ending Node
        :type to_node: str

        :param from_column: the list of columns located in the starting Node in order, these columns
            will be connected to the ending Node columns
        :type from_column: list[str]

        :param to_column: the list of columns located in the ending Node
        :type to_column: list[str]

        **Example**

        .. code-block :: python

            data_join_graph.edge(from_node="main", to_node="transactions",
                                 from_column=["transaction_id"], to_column=["transaction_id"])
        """
        self.node_edges[from_node] |= set(from_column)
        self.node_edges[to_node] |= set(to_column)

        self.edges.append(Edge(from_node=from_node, to_node=to_node, from_column=from_column, to_column=to_column))

    def _visualize_nodes(self):
        for node in self.nodes:
            name = node.table.name
            column_names = self.node_edges[name]
            self.edges_to_ports.update({f"{name}{column}": f"{name}:e{i}" for i, column in enumerate(column_names)})
            columns = ''.join(
                [self.ColumnTemplate.format(port=f"e{i}", column_name=name) for i, name in enumerate(column_names)])
            node_label = self.NodeTemplate[:self.ColumnsIndex] + columns + self.NodeTemplate[self.ColumnsIndex:]
            color = "LIGHTBLUE2" if self.main_node_name == name else "WHITE"

            super().node(name=name, label=node_label.format(color=color,
                                                            dataset_name=name,
                                                            colspan=str(len(column_names))))

    def _visualize_edges(self):
        for edge in self.edges:
            edges = [(self.edges_to_ports[edge.from_node + from_col], self.edges_to_ports[edge.to_node + to_col])
                     for from_col, to_col in zip(edge.from_column, edge.to_column)]

            super().edges(edges)

    @check_graphviz_binaries
    def visualize(self):
        super().clear()
        super().attr('node', shape='plaintext')
        super().attr('edge', minlen='2')
        super().attr('graph', nodesep='1')

        self._visualize_nodes()
        self._visualize_edges()

        """Display graph in the notebook or as a rendered image."""
        if is_ipython():
            import IPython.display
            IPython.display.display(self)

        else:
            self.render(view=True)

    def to_dict(self):
        """Convert this Node to dictionary for further REST API call."""
        _dict = {
            "id": "obm",
            "type": "execution_node",
            "op": "kube",
            "runtime_ref": "obm",
            "inputs": [{"id": node.table.name} for node in self.nodes],
            "outputs": [
                {
                    "id": "obm_out"
                }
            ],
            "parameters": {
                "stage_flag": True,
                "output_logs": True,
                "engine": {
                    "template_id": "spark-3.2-automl-template",
                },
                "obm": {
                    "Entity_Graph": {
                        "nodes": [node.to_dict() for node in self.nodes],
                        "edges": [edge.to_dict() for edge in self.edges]
                    },
                    "Tables": {node.table.name: node.table.to_dict() for node in self.nodes},
                    "Feature_Selector": {
                        "selectors": [
                            "deduplicate",
                            "consistent"
                        ]
                    },
                    "OneButtonMachine": {
                        "main_table": self.main_node_name,
                        "target_column": self.target_column,
                        "max_depth": self.max_depth,
                        "data_source": self.data_source_type,
                        "problem_type": self.problem_type,
                        "join_limit": 50,
                    }
                }
            }
        }

        if self.fairness_info is not None:
            _dict['parameters']['fairness_info'] = self.fairness_info

        return _dict

    @classmethod
    def _from_dict(cls, _dict: dict) -> 'DataJoinGraph':
        """Create data join graph object from wml pipeline parameters."""
        data_join_graph = cls()

        for node in _dict['parameters']['obm']['Entity_Graph']['nodes']:
            if node['table_name'] == _dict['parameters']['obm']['OneButtonMachine']['main_table']:
                data_join_graph.node(name=node['table_name'], main=True)

            else:
                data_join_graph.node(name=node['table_name'])

        [data_join_graph.edge(
            from_node=edge['from'],
            to_node=edge['to'],
            from_column=edge['from_column'],
            to_column=edge['to_column']) for edge in _dict['parameters']['obm']['Entity_Graph']['edges']]

        data_join_graph.fairness_info = _dict['parameters'].get('fairness_info')

        return data_join_graph

    def pretty_print(self, show_imports: bool = True, ipython_display: bool = False) -> Union[str, None]:
        """Get the Python source code representation of the DataJoinGraph.

        :param show_imports: whether to include import statements in the pretty-printed code
        :type show_imports: bool, optional

        :param ipython_display:
            - `False` - return the pretty-printed code as a plain old Python string
            - `True` - pretty-print in notebook cell output with syntax highlighting
            - 'input' - create a new notebook cell with pretty-printed code as input

        :type ipython_display: bool or str

        :return: if called with `ipython_display=False`, return pretty-printed Python source code as a Python string
        :rtype: str or None
        """
        nodes_str = ""
        for node in self.nodes:
            if self.main_node_name == node.table.name:
                nodes_str = nodes_str + node._pretty_print(main=True)

            else:
                nodes_str = nodes_str + node._pretty_print()

        edges_str = ""
        for edge in self.edges:
            edges_str = edges_str + edge._pretty_print()

        result = f"""{'from ibm_watson_machine_learning.preprocessing import DataJoinGraph' if show_imports else ''}
data_join_graph = DataJoinGraph()
{nodes_str}
{edges_str}
"""

        if ipython_display == False:
            return result

        elif ipython_display == 'input':
            import IPython.core
            ipython = IPython.core.getipython.get_ipython()
            comment = "# generated by pretty_print(ipython_display='input') from previous cell\n"
            ipython.set_next_input(comment + result, replace=False)

        else:
            assert ipython_display in [True, 'output']
            import IPython.display
            markdown = IPython.display.Markdown(f'```python\n{result}\n```')
            return IPython.display.display(markdown)

    def __str__(self):
        return (f"\nDataJoinGraph:\n"
                f"\tMAIN NODE: {self.main_node_name}\n"
                f"\tTARGET COLUMN: {self.target_column}\n"
                f"\tMAX DEPTH: {self.max_depth}\n"
                f"\tDATA SOURCE TYPE: {self.data_source_type}\n"
                f"\tPROBLEM TYPE: {self.problem_type}\n"
                f"\tNODES: {[node for node in self.nodes]}\n"
                f"\tEDGES: {[edge for edge in self.edges]}"
                )
