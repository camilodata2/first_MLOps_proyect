__all__ = [
    'DataJoinPipeline',
    'OBMPipelineGraphBuilder'
]

#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import ast
import re
from operator import itemgetter
from collections import defaultdict
from typing import Tuple, List, Union, Dict, TYPE_CHECKING
from graphviz import Digraph
from .templates import pretty_print_template
from ..utils.autoai.utils import is_ipython, check_graphviz_binaries, try_import_lale
from ..utils.autoai.enums import VisualizationTypes

if TYPE_CHECKING:
    from ..helpers.connections.connections import DataConnection


class DataJoinPipeline:
    """Class representing abstract data join pipeline."""
    VisualizationTypes = VisualizationTypes

    def __init__(self, preprocessed_data_connection: 'DataConnection', optimizer: 'RemoteAutoPipelines') -> None:
        """
        :param preprocessed_data_connection: populated DataConnection with preprocessed data join information
        :type preprocessed_data_connection: DataConnection

        :param optimizer: optimizer with initialized pipeline metadata, required for running an OBM training
            in predict method
        :type optimizer: RemoteAutoPipeline
        """
        self._obm_json = preprocessed_data_connection._download_obm_json()
        self._pipeline_json = self._obm_json['Pipeline']
        self._obm_pipeline_graph_builder = OBMPipelineGraphBuilder(self._obm_json)
        self._graph_json, self._graph = self._obm_pipeline_graph_builder.build_graph()
        try_import_lale()
        self.lale_pipeline = self._obm_pipeline_graph_builder.build_preprocessing_prefix()
        self._optimizer = optimizer

    @check_graphviz_binaries
    def visualize(self, astype: 'VisualizationTypes' = VisualizationTypes.INPLACE) -> None:
        """Display graph in the notebook or as a rendered image.

        :param astype: specify a type of visualization for this graph, default is VisualizationTypes.INPLACE
            (when in notebook env, picture will be displayed in the cell output),
            VisualizationTypes.PDF indicates to render a pdf document with visualization
        :type astype: VisualizationTypes, optional
        """
        if hasattr(self, 'lale_pipeline'):
            self.lale_pipeline.visualize()
        else:
            if is_ipython():
                if astype == VisualizationTypes.PDF:
                    self._graph.render(view=True)

                else:
                    import IPython.display
                    IPython.display.display(self._graph)

            else:
                self._graph.render(view=True)

    # def predict(self,
    #             *,
    #             training_data_reference: List['DataConnection'],
    #             training_results_reference: 'DataConnection' = None,
    #             background_mode=False) -> 'DataFrame':
    #     """
    #     Run an OBM training process on top of the training data referenced by DataConnection.
    #
    #     Parameters
    #     ----------
    #     training_data_reference: List[DataConnection], required
    #         Data storage connection details to inform where training data is stored.
    #
    #     training_results_reference: DataConnection, optional
    #         Data storage connection details to store pipeline training results. Not applicable on CP4D.
    #
    #     background_mode: bool, optional
    #         Indicator if fit() method will run in background (async) or (sync).
    #
    #     Returns
    #     -------
    #     pandas.DataFrame contains dataset from remote data storage.
    #     """
    #     self._optimizer.fit(training_data_reference=training_data_reference,
    #                         training_results_reference=training_results_reference,
    #                         background_mode=background_mode)
    #
    #     return self._optimizer.get_preprocessed_data_connection().read()

    def pretty_print(self, ipython_display: bool = False) -> None:
        """Print code which generates OBM data preprocessing.

        :param ipython_display: if method executed in jupyter notebooks/ipython set this flag to true in order
            to get syntax highlighting
        :type ipython_display: bool, optional
        """
        if hasattr(self, 'lale_pipeline'):
            if ipython_display:
                self.lale_pipeline.pretty_print(ipython_display=True)
            else:
                print(self.lale_pipeline.pretty_print())
        else:
            params = self._optimizer.get_params()
            data_join_graph = params["data_join_graph"]

            nodes = ""
            for node in data_join_graph.nodes:
                if node.table.timestamp_format and node.table.timestamp_column_name:
                    nodes += f"data_join_graph.node(name=\"{node.table.name}\"," \
                             f" timestamp_column_name=\"{node.table.timestamp_column_name}\"," \
                             f" timestamp_format=\"{node.timestamp_format}\")\n"
                else:
                    nodes += f"data_join_graph.node(name=\"{node.table.name}\")\n"

            edges = "\n".join([f"data_join_graph.edge(from_node=\"{edge.from_node}\", to_node=\"{edge.to_node}\",\n"
                               f"\t\t\t\t\t from_column={edge.from_column}, to_column={edge.to_column})"
                               for edge in data_join_graph.edges])

            join_indices = [it - 1 for it in self._obm_pipeline_graph_builder.get_join_iterations()]
            paths = "\n".join(["# - " + re.search(r"\[(.*)\]", msg["message"]["text"]).group()
                               for msg in itemgetter(*join_indices)(self._pipeline_json)])
            nb_of_features = re.search(r"\d", self._pipeline_json[-1]["message"]["text"]).group()

            pretty_print = pretty_print_template.template.format(
                nodes=nodes, edges=edges,
                name=f"\"{params['name']}\"",
                prediction_type=f"\"{params['prediction_type']}\"",
                prediction_column=f"\"{params['prediction_column']}\"",
                scoring=f"\"{params['scoring']}\"",
                paths=paths,
                nb_of_features=nb_of_features)

            if ipython_display:
                import IPython.display
                markdown = IPython.display.Markdown(f'```python\n{pretty_print}\n```')
                IPython.display.display(markdown)
            else:
                print(pretty_print)


class OBMPipelineGraphBuilder:
    """Class for extracting particular elements from OBM output json with pipeline description.

    :param pipeline_json: dictionary with loaded obm.json file
    :type pipeline_json: dict
    """

    def __init__(self, obm_json: dict) -> None:
        self.pipeline_json = obm_json['Pipeline']
        self.trained_pipeline_json = obm_json['OneButtonMachine']
        self.tables_json = obm_json['Tables']
        self.categorical_column_names = self.get_categorical_columns(obm_json)
        self.main_table = self.get_main_table_name(obm_json)
        self.main_table_timestamp = self.get_time_column_for_table(self.tables_json, self.main_table)
        self.group_by_key = self.get_group_by_key(self.tables_json, self.main_table)
        self.last_non_join_iteration = self.get_last_non_join_iteration()
        self.selection_iteration = self.get_selection_iteration()
        self.join_iterations = self.get_join_iterations()
        self.tables_info = self.get_tables_info(self.tables_json)
        self.graph_json = {'nodes': [], 'edges': defaultdict(set)}
        self.graph = Digraph(comment='Data Preprocessing Steps Graph',
                             node_attr={'color': 'lightblue2', 'style': 'filled'})

    @staticmethod
    def get_step_details(msg_json) -> Tuple[str, int, str]:
        """Get particular step name, iteration number and step description."""
        name = msg_json['feature_engineering_components']['obm'][0]['step_name'].split(':')[1]
        iteration = msg_json['feature_engineering_components']['obm'][0]['iteration']
        text = msg_json['message']['text']
        return name, iteration, text

    def get_step_types(self) -> List[str]:
        """For all steps get their types."""
        return [message['feature_engineering_components']['obm'][0]['step_type'] for message in self.pipeline_json]

    def get_last_non_join_iteration(self) -> int:
        """Get a number of the last step before join."""
        return self.get_step_types().index('join')

    def get_selection_iteration(self) -> int:
        """Get feature selection step number."""
        return self.get_step_types().index('feature selection') + 1

    def get_join_iterations(self) -> List[int]:
        """Get list of join step numbers."""
        steps_types = [message['feature_engineering_components']['obm'][0]['step_type'] for message in
                       self.pipeline_json]
        return [i + 1 for i, x in enumerate(steps_types) if x == "join"]

    @staticmethod
    def get_join_extractors(msg_json: dict) -> Union[dict, None]:
        """Get extractors if exist."""
        return msg_json['feature_engineering_components']['obm'][0].get('extractors')

    @staticmethod
    def get_extractor_columns(extractor_json: dict) -> List[str]:
        """Get all columns names from particular extractor."""
        return extractor_json['columns']

    @staticmethod
    def get_extractor_transformations(extractor_json: dict) -> Dict[str, str]:
        """Get a dictionary with all transformations names."""
        return extractor_json['transformations']

    @staticmethod
    def get_join_info(msg_json: dict) -> Tuple[List[str], List[List[str]]]:
        """Get join_tables and keys if exist."""
        join_path = msg_json['feature_engineering_components']['obm'][0].get('join_path')
        if join_path is not None:
            return join_path.get("tables"), join_path.get("join_keys")
        else:
            return None, None

    @staticmethod
    def get_tables_info(table_json: dict) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get information about all tables such as name and a list of column names if exists."""
        tables_info = {}
        for _, table in enumerate(table_json):
            column_names = [name for _, name in enumerate(table_json.get(table).get('training_column_types'))]
            column_formats = table_json.get(table).get('column_format')
            tables_info[table] = (column_names, column_formats)
        return tables_info

    @staticmethod
    def get_main_table_name(obm_json: dict) -> str:
        """Get the name of the main table for OneBM"""
        return obm_json.get("OneButtonMachine").get("main_table")

    @staticmethod
    def get_time_column_for_table(tables_json: dict, table_name: str) -> str:
        """Get the name of the timestamp column for the table. """
        return tables_json.get(table_name).get("timestamp_column_name")

    @staticmethod
    def get_group_by_key(tables_json, main_table) -> str:
        """Get the name of the primary key for the main table if present.
        This key is used for all group by operations after joins. If there is no primary key,
        row_id of the main table is used internally."""
        main_table_info = tables_json.get(main_table)
        return main_table_info.get("primary_key")

    @staticmethod
    def get_feature_info(feature_info_json: dict) -> dict:
        join_path = feature_info_json.get("join_path")
        results = {}
        results['tables'] = join_path.get("tables")
        results['join_keys'] = join_path.get("join_keys")
        results['feature_name'] = feature_info_json.get("feature_name")
        results['op'] = feature_info_json.get("expression", {}).get("op")
        results['params'] = feature_info_json.get("expression", {}).get("params")
        results['pattern'] = feature_info_json.get("pattern")
        results['timestamp_condition'] = feature_info_json.get("timestamp_condition")
        results['join_limit'] = feature_info_json.get("join_limit")
        results['sliding_window_length'] = feature_info_json.get("sliding_window_length")
        return results

    @staticmethod
    def get_categorical_columns(obm_json: dict) -> List[str]:
        return obm_json.get("Category").get("categorical_column_names")

    def build_extractors_subgraph(self, msg_json: dict, join_iteration: int) -> None:
        """Create sub-graph for extractors representation."""
        extractors = self.get_join_extractors(msg_json)
        join_iteration = str(join_iteration)

        if extractors is not None:
            for ext, i in zip(extractors, range(len(extractors))):
                self.graph.attr('node', color='lightgray')
                ext_index = join_iteration + str(i)

                self.graph.node(ext_index, ext)
                self.graph.edge(join_iteration, ext_index)
                self.graph_json['nodes'].append(ext_index)
                self.graph_json['edges'][join_iteration].add(ext_index)

                columns = self.get_extractor_columns(extractors[ext])
                transformations = self.get_extractor_transformations(extractors[ext])

                for j, column in enumerate(columns):
                    self.graph.attr('node', color='lightgreen')
                    col_index = join_iteration + str(i) + str(j)

                    self.graph.node(col_index, column)
                    self.graph.edge(ext_index, col_index)
                    self.graph_json['nodes'].append(col_index)
                    self.graph_json['edges'][ext_index].add(col_index)

                    self.graph.attr('node', color='lightblue2')
                    for transformation in transformations:
                        self.graph.edge(col_index, transformation)
                        self.graph.edge(transformation, str(self.selection_iteration))
                        self.graph_json['edges'][col_index].add(transformation)
                        self.graph_json['edges'][transformation].add(str(self.selection_iteration))

    def build_preprocessing_prefix(self) -> 'lale.operators.TrainableIndividualOp':
        """Create a Lale pipeline corresponding to the data join pipeline. This pipeline has all the information to
        create a pipeline that can be used to perform transform, but not for fit."""
        from lale.expressions import (Expr, it, replace, sum, max, count, day_of_month,
                                      day_of_week, day_of_year, month, hour, minute, min,
                                      distinct_count, mean, variance, item, recent, window_max,
                                      window_max_trend, window_mean, window_mean_trend, window_min,
                                      window_min_trend, window_variance, window_variance_trend,
                                      normalized_count, normalized_sum, trend, recent_gap_to_cutoff,
                                      max_gap_to_cutoff, string_indexer)
        from lale.lib.lale import Scan, Join, Map, GroupBy, Aggregate, ConcatFeatures, Project, Relational
        from lale.operators import make_pipeline_graph

        transformation_to_func_map = {"max": max, "count": count, "sum": sum, "day_of_month": day_of_month,
                                      "day_of_week": day_of_week, "day_of_year": day_of_year, "month": month,
                                      "hour": hour, "minute": minute,
                                      "min": min, "mean": mean, "variance": variance, "distinct_count": distinct_count,
                                      "ItemsetMI": item, "ItemsetCOR": item,
                                      "SymbolRecent": recent, "recent": recent, "window_max": window_max,
                                      "SymbolRecent": recent, "window_max": window_max,
                                      "window_max_trend": window_max_trend,
                                      "window_mean": window_mean, "window_mean_trend": window_mean_trend,
                                      "window_min": window_min,
                                      "window_min_trend": window_min_trend, "window_variance": window_variance,
                                      "window_variance_trend": window_variance_trend,
                                      "normalized_count": normalized_count, "normalised_count": normalized_count,
                                      "normalized_sum": normalized_sum, "normalised_sum": normalized_sum, "trend": trend,
                                      "recent_gap_to_cutoff": recent_gap_to_cutoff,
                                      "max_gap_to_cutoff": max_gap_to_cutoff}
        time_series_functions = ["day_of_month", "day_of_week", "day_of_year", "minute", "hour", "month"]
        parameterized_functions = ["SymbolRecent", "recent", "window_max",
                                   "window_max_trend", "window_mean", "window_mean_trend", "window_min",
                                   "window_min_trend", "window_variance", "window_variance_trend",
                                   "recent_gap_to_cutoff"]
        time_cufoff_functions = ["max_gap_to_cutoff", "recent_gap_to_cutoff"]
        level_3_nodes_list = []
        joins = {}
        for msg in self.trained_pipeline_json.get("feature_engineering_components")[0].get("obm"):
            feature_info = self.get_feature_info(msg)
            join_tables = feature_info["tables"]
            join_keys = feature_info["join_keys"]
            feature_name = feature_info["feature_name"]
            op = feature_info["op"]
            column = feature_info["params"]
            pattern = feature_info["pattern"]
            timestamp_condition = feature_info["timestamp_condition"]
            join_limit = feature_info["join_limit"]
            sliding_window_length = feature_info["sliding_window_length"]
            if join_tables is None or len(join_tables) == 0:
                join_tables = [self.main_table]
            join_tables_str = "_".join(join_tables)
            _, _, feature_list = joins.get(join_tables_str, (None, None, []))
            feature_list.append(
                (feature_name, op, column, pattern, timestamp_condition, join_limit, sliding_window_length))
            joins[join_tables_str] = (join_tables, join_keys, feature_list)
        scan_tables_already_added = {}
        #The pipeline graph needs to be built level-wise, so keep track of steps and edges.
        pipeline_steps = []
        pipeline_edges = []
        for join_tables_str, join_info in joins.items():
            scan_table_nodes = []
            join_keys_idx = 0
            predicates_list = []
            join_tables, join_keys, feature_list = join_info
            if feature_list is not None and len(feature_list) > 0:
                # Get the time_condition, join_limit and sliding_window_length from the first feature for this join.
                # The assumption is that these fields exist for all the features for the join where applicable.
                elem1 = feature_list[0]
                feature_name, op, column, pattern, timestamp_condition, join_limit, sliding_window_length = elem1
            for table_idx, table in enumerate(join_tables):
                scan_table_node = scan_tables_already_added.get(table, None)
                if table not in scan_tables_already_added:
                    scan_table_node = Scan(
                        table=Expr(ast.Subscript(value=ast.Name('it', ast.Store()), slice=ast.Index(ast.Str(s=table)))))
                    pipeline_steps.append(scan_table_node)
                    scan_tables_already_added[table] = scan_table_node
                scan_table_nodes.append(scan_table_node)
                # create a join predicate for this table and next if this is not the last table
                if table_idx != len(join_tables) - 1:
                    next_table = join_tables[table_idx + 1]
                    keys_for_this_table = join_keys[join_keys_idx]
                    join_keys_idx += 1
                    keys_for_next_table = join_keys[join_keys_idx]
                    join_keys_idx += 1
                    for key_idx in range(len(keys_for_this_table)):
                        lhs = ast.Subscript(
                            value=ast.Subscript(value=ast.Name('it', ast.Store()), slice=ast.Index(ast.Str(s=table))),
                            slice=ast.Index(ast.Str(s=keys_for_this_table[key_idx])))
                        rhs = ast.Subscript(
                            value=ast.Subscript(value=ast.Name('it', ast.Store()),
                                                slice=ast.Index(ast.Str(s=next_table))),
                            slice=ast.Index(ast.Str(s=keys_for_next_table[key_idx])))
                        predicates_list.append(Expr(ast.Compare(left=lhs, ops=[ast.Eq()], comparators=[rhs])))
            # Adding a timestamp based predicate based on the timestamp_condition
            if timestamp_condition is not None:
                # timestamp_condition element i contains the table name and element i+1 contains the column name
                lhs = ast.Subscript(
                    value=ast.Subscript(value=ast.Name('it', ast.Store()),
                                        slice=ast.Index(ast.Str(s=timestamp_condition[0]))),
                    slice=ast.Index(ast.Str(s=timestamp_condition[1])))
                rhs = ast.Subscript(
                    value=ast.Subscript(value=ast.Name('it', ast.Store()),
                                        slice=ast.Index(ast.Str(s=timestamp_condition[2]))),
                    slice=ast.Index(ast.Str(s=timestamp_condition[3])))
                predicates_list.append(Expr(ast.Compare(left=lhs, ops=[ast.GtE()], comparators=[rhs])))

            # Assuming each extractor covers only one join
            if len(predicates_list) > 0:
                join_node = Join(pred=predicates_list, join_limit=join_limit,
                                 sliding_window_length=sliding_window_length)
                pipeline_steps.append(join_node)
                for i in range(0, len(scan_table_nodes)):
                    pipeline_edges.append((scan_table_nodes[i], join_node))
            else:
                join_node = None

            aggregate_transformations = {}
            map_transformations = {}
            column_projections = {}
            for ext, i in zip(feature_list, range(len(feature_list))):
                feature_name, op, params, pattern, _, _, _ = ext
                column = None if params is None else params[0][
                                                     1:]  # The first param is a column name, and starts with '$'
                if op is None or op == "identity":
                    # if op is identity, column info exists, if it is None, assume that it is
                    # the column of the main table with the same name as feature_name
                    if column is None:
                        column = feature_name
                    column_projections[feature_name] = Expr(
                        ast.Subscript(value=ast.Name('it', ast.Store()), slice=ast.Index(ast.Str(s=column))))
                else:
                    transformation_func = transformation_to_func_map.get(op)
                    if transformation_func is None:
                        # Check for prefix match, such as for SymbolRecent_0 etc.
                        transformation_func = next(
                            (v for k, v in transformation_to_func_map.items() if k == (re.split('(\\d+)', op)[0][:-1])),
                            None)
                    if transformation_func is not None:
                        if op in time_series_functions:
                            # these functions need time format of the column as input which is obtained from the tables info
                            time_formats = [self.tables_info.get(table)[1].get(column) for table in join_tables]
                            time_format = next((item for item in time_formats if item is not None), "None")
                            map_transformations[feature_name] = transformation_func(
                                Expr(ast.Subscript(value=ast.Name('it', ast.Store()),
                                                   slice=ast.Index(ast.Str(s=column)))),
                                fmt=time_format)
                        else:
                            kwargs = {}
                            if op in ["ItemsetMI", "ItemsetCOR"]:
                                if pattern is not None:
                                    kwargs['value'] = pattern
                                else:
                                    kwargs['value'] = ""
                                    print("Pattern missing for ItemsetMI or ItemsetCOR")  # TODO: Raise an exception
                            if re.split('(\\d+)', op)[0][:-1] in parameterized_functions:
                                if op.startswith("window"):
                                    kwargs['size'] = int(re.split('(\\d+)', op)[1])
                                else:
                                    kwargs['age'] = int(re.split('(\\d+)', op)[1])
                            if (re.split('(\\d+)', op)[0][:-1] in time_cufoff_functions or
                                re.split('(\\d+)', op)[0] in time_cufoff_functions):
                                kwargs['cutoff'] = Expr(
                                    ast.Subscript(value=ast.Name(self.main_table, ast.Store()), slice=ast.Index(
                                        ast.Str(s=self.main_table_timestamp))))
                            aggregate_transformations[feature_name] = transformation_func(
                                Expr(ast.Subscript(value=ast.Name('it', ast.Store()),
                                                   slice=ast.Index(ast.Str(s=column)))), **kwargs)
                    else:
                        print("Unknown transformation: ", op)
            ops_list = []
            if len(column_projections) > 0:
                # Map is an extended projection operation, so combining map and projects here.
                map_transformations.update(column_projections)
            if len(map_transformations) > 0:
                ops_list.append(Map(columns=map_transformations, remainder="drop"))

            if len(aggregate_transformations) > 0:
                if self.group_by_key is None:
                    group_by_expr = Expr(
                        ast.Subscript(value=ast.Name('it', ast.Store()), slice=ast.Index(ast.Str(s='row_id'))))
                elif isinstance(self.group_by_key, str):
                    group_by_expr = Expr(
                        ast.Subscript(value=ast.Name('it', ast.Store()), slice=ast.Index(ast.Str(s=self.group_by_key))))
                elif isinstance(self.group_by_key, list):
                    group_by_expr = [
                        Expr(ast.Subscript(value=ast.Name('it', ast.Store()), slice=ast.Index(ast.Str(s=column_name))))
                        for column_name in self.group_by_key]
                ops_list.append(Aggregate(columns=aggregate_transformations, group_by=group_by_expr))
            if len(ops_list) > 0:
                for i in range(0, len(ops_list)):
                    pipeline_steps.append(ops_list[i])
                    if join_node is not None:
                        pipeline_edges.append((join_node, ops_list[i]))
                    else:
                        for i in range(0, len(scan_table_nodes)):
                            pipeline_edges.append((scan_table_nodes[i], ops_list[i]))
                level_3_nodes_list.extend(ops_list)
            elif join_node is not None: #if ops_list is empty, the nodes from the last level in the graph are either scan or join
                level_3_nodes_list.append(join_node)
            else:
                level_3_nodes_list.extend(scan_table_nodes)
        # Use ConcatFeatures on the extractors_prefix and then selectFeatures
        concat = ConcatFeatures()
        pipeline_steps.append(concat)
        for i in range(0, len(level_3_nodes_list)):
            pipeline_edges.append((level_3_nodes_list[i], concat))

        # Add the component that converts categorical columns to numerical using a string indexer.
        if self.categorical_column_names is not None:
            string_indexer_encoding = Map(columns=[
                string_indexer(
                    ast.Subscript(value=ast.Name('it', ast.Store()), slice=ast.Index(ast.Str(s=categorical_column))))
                for categorical_column in self.categorical_column_names], remainder="passthrough")
            pipeline_steps.append(string_indexer_encoding)
            pipeline_edges.append((concat, string_indexer_encoding))
        nested_pipeline = make_pipeline_graph(pipeline_steps, pipeline_edges)
        return Relational(operator=nested_pipeline)

    def build_graph(self) -> Tuple[dict, 'Digraph']:
        """Create a graphviz Digraph with pipeline representation."""
        for msg in self.pipeline_json:
            name, iteration, text = self.get_step_details(msg)
            self.graph.node(str(iteration), name, tooltip=text)
            self.graph_json['nodes'].append(str(iteration))

            if 1 < iteration <= self.last_non_join_iteration:
                self.graph.edge(str(iteration - 1), str(iteration))
                self.graph_json['edges'][str(iteration - 1)].add(str(iteration))
            elif iteration in self.join_iterations:
                self.graph.edge(str(self.last_non_join_iteration), str(iteration))
                self.graph_json['edges'][str(self.last_non_join_iteration)].add(str(iteration))
                self.build_extractors_subgraph(msg, iteration)
            elif iteration > self.selection_iteration:
                self.graph.edge(str(iteration - 1), str(iteration))
                self.graph_json['edges'][str(iteration - 1)].add(str(iteration))

        return self.graph_json, self.graph
