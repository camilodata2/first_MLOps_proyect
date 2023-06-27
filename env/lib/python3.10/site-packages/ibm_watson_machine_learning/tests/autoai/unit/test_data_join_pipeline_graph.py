#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import json
from ibm_watson_machine_learning.preprocessing.data_join_pipeline import OBMPipelineGraphBuilder


baseline = ['// Data join graph', 'digraph {', 'node [color=lightblue2 style=filled]',
                '1 [label=" Parsed problem description" tooltip=""]', '2 [label=" Loading data" tooltip=""]', '1 -> 2',
                '3 [label=" Feature extraction" tooltip="Extracting features from the path:[main]. Checking 1 over 5 paths. Found 10 features."]',
                '2 -> 3', 'node [color=lightgray]', '30 [label=Identity]', '3 -> 30', 'node [color=lightgreen]',
                '300 [label=group_id]', '30 -> 300', 'node [color=lightblue2]', '300 -> Identity', 'Identity -> 8',
                'node [color=lightgreen]', '301 [label=group_customer_id]', '30 -> 301', 'node [color=lightblue2]',
                '301 -> Identity', 'Identity -> 8', 'node [color=lightgreen]', '302 [label=transaction_id]',
                '30 -> 302', 'node [color=lightblue2]', '302 -> Identity', 'Identity -> 8', 'node [color=lightgreen]',
                '303 [label=comments]', '30 -> 303', 'node [color=lightblue2]', '303 -> Identity', 'Identity -> 8',
                'node [color=lightgray]', '31 [label=TimeStamp]', '3 -> 31', 'node [color=lightgreen]',
                '310 [label=prefix_0_time]', '31 -> 310', 'node [color=lightblue2]', '310 -> day_of_week',
                'day_of_week -> 8', '310 -> minute', 'minute -> 8', '310 -> hour', 'hour -> 8', '310 -> day_of_month',
                'day_of_month -> 8', '310 -> day_of_year', 'day_of_year -> 8', '310 -> month', 'month -> 8',
                '4 [label=" Feature extraction" tooltip="Extracting feature from the path:[main](group_id)[purchases]. Checking 2 over 5 paths. Found 28 features."]',
                '2 -> 4', 'node [color=lightgray]', '40 [label=TimeStampSeriesCutoff]', '4 -> 40',
                'node [color=lightgreen]', '400 [label=time]', '40 -> 400', 'node [color=lightblue2]',
                '400 -> max_gap_to_cutoff', 'max_gap_to_cutoff -> 8', '400 -> count', 'count -> 8',
                '400 -> normalised_count', 'normalised_count -> 8', '400 -> recent_gap_to_cutoff',
                'recent_gap_to_cutoff -> 8', 'node [color=lightgray]', '41 [label=TimeSeriesCutoff]', '4 -> 41',
                'node [color=lightgreen]', '410 [label=price]', '41 -> 410', 'node [color=lightblue2]', '410 -> trend',
                'trend -> 8', '410 -> variance', 'variance -> 8', '410 -> recent', 'recent -> 8',
                '410 -> sliding_window', 'sliding_window -> 8', '410 -> mean', 'mean -> 8', '410 -> min', 'min -> 8',
                '410 -> max', 'max -> 8', '410 -> normalised_sum', 'normalised_sum -> 8', '410 -> sum', 'sum -> 8',
                '5 [label=" Feature extraction" tooltip="Extracting feature from the path:[main](transaction_id)[transactions]. Checking 3 over 5 paths. Found 13 features."]',
                '2 -> 5', 'node [color=lightgray]', '50 [label=NumberSet]', '5 -> 50', 'node [color=lightgreen]',
                '500 [label=product_id]', '50 -> 500', 'node [color=lightblue2]', '500 -> count', 'count -> 8',
                '500 -> variance', 'variance -> 8', '500 -> mean', 'mean -> 8', '500 -> min', 'min -> 8', '500 -> max',
                'max -> 8', '500 -> sum', 'sum -> 8', 'node [color=lightgray]', '51 [label=ItemSet]', '5 -> 51',
                'node [color=lightgreen]', '510 [label=description]', '51 -> 510', 'node [color=lightblue2]',
                '510 -> count', 'count -> 8', '510 -> distinct_count', 'distinct_count -> 8', 'node [color=lightgray]',
                '52 [label=ItemSetPattern]', '5 -> 52', 'node [color=lightgreen]', '520 [label=description]',
                '52 -> 520', 'node [color=lightblue2]', '520 -> item', 'item -> 8',
                '6 [label=" Feature extraction" tooltip="Extracting feature from the path:[main](transaction_id)[transactions](product_id)[products]. Checking 4 over 5 paths. Found 11 features."]',
                '2 -> 6', 'node [color=lightgray]', '60 [label=NumberSet]', '6 -> 60', 'node [color=lightgreen]',
                '600 [label=price]', '60 -> 600', 'node [color=lightblue2]', '600 -> count', 'count -> 8',
                '600 -> variance', 'variance -> 8', '600 -> mean', 'mean -> 8', '600 -> min', 'min -> 8', '600 -> max',
                'max -> 8', '600 -> sum', 'sum -> 8', 'node [color=lightgray]', '61 [label=ItemSet]', '6 -> 61',
                'node [color=lightgreen]', '610 [label=type]', '61 -> 610', 'node [color=lightblue2]', '610 -> count',
                'count -> 8', '610 -> distinct_count', 'distinct_count -> 8', 'node [color=lightgray]',
                '62 [label=ItemSetPattern]', '6 -> 62', 'node [color=lightgreen]', '620 [label=type]', '62 -> 620',
                'node [color=lightblue2]', '620 -> item', 'item -> 8',
                '7 [label=" Feature extraction" tooltip="Extracting feature from the path:[main](group_customer_id)[customers]. Checking 5 over 5 paths. Found 22 features."]',
                '2 -> 7', 'node [color=lightgray]', '70 [label=ItemSet]', '7 -> 70', 'node [color=lightgreen]',
                '700 [label=name]', '70 -> 700', 'node [color=lightblue2]', '700 -> count', 'count -> 8',
                '700 -> distinct_count', 'distinct_count -> 8', 'node [color=lightgreen]', '701 [label=address]',
                '70 -> 701', 'node [color=lightblue2]', '701 -> count', 'count -> 8', '701 -> distinct_count',
                'distinct_count -> 8', 'node [color=lightgray]', '71 [label=NumberSet]', '7 -> 71',
                'node [color=lightgreen]', '710 [label=age]', '71 -> 710', 'node [color=lightblue2]', '710 -> count',
                'count -> 8', '710 -> variance', 'variance -> 8', '710 -> mean', 'mean -> 8', '710 -> min', 'min -> 8',
                '710 -> max', 'max -> 8', '710 -> sum', 'sum -> 8', 'node [color=lightgreen]',
                '711 [label=number_children]', '71 -> 711', 'node [color=lightblue2]', '711 -> count', 'count -> 8',
                '711 -> variance', 'variance -> 8', '711 -> mean', 'mean -> 8', '711 -> min', 'min -> 8', '711 -> max',
                'max -> 8', '711 -> sum', 'sum -> 8', 'node [color=lightgreen]', '712 [label=income]', '71 -> 712',
                'node [color=lightblue2]', '712 -> count', 'count -> 8', '712 -> variance', 'variance -> 8',
                '712 -> mean', 'mean -> 8', '712 -> min', 'min -> 8', '712 -> max', 'max -> 8', '712 -> sum',
                'sum -> 8', 'node [color=lightgray]', '72 [label=ItemSetPattern]', '7 -> 72', 'node [color=lightgreen]',
                '720 [label=name]', '72 -> 720', 'node [color=lightblue2]', '720 -> item', 'item -> 8',
                'node [color=lightgreen]', '721 [label=address]', '72 -> 721', 'node [color=lightblue2]', '721 -> item',
                'item -> 8',
                '8 [label=" Feature selection" tooltip=" Found 66 features after feature selection with deduplicate consistent"]',
                '9 [label=" Output feature and model" tooltip="Writing 66 features"]', '8 -> 9', '}']


def compare_graph_to_baseline(g):
    list_str_g = g.__str__().split('\n')
    return [line.strip() for line in list_str_g] == baseline

class TestOBMGraph(unittest.TestCase):
    obm_json_path = './autoai/artifacts/group_customer/obm.json'
    pipeline_json = None
    builder = None

    @classmethod
    def setUp(cls) -> None:
        with open(cls.obm_json_path) as json_file:
            cls.obm_json = json.load(json_file)
            cls.pipeline_json = cls.obm_json['Pipeline']

        assert len(TestOBMGraph.pipeline_json) > 0

    def test_01_initialize_graph_object(self):
        TestOBMGraph.builder = OBMPipelineGraphBuilder(TestOBMGraph.obm_json)

        assert isinstance(TestOBMGraph.builder, OBMPipelineGraphBuilder)
        assert TestOBMGraph.pipeline_json == TestOBMGraph.builder.pipeline_json
        assert TestOBMGraph.builder.get_last_non_join_iteration() == 2
        assert TestOBMGraph.builder.get_selection_iteration() == 8
        assert TestOBMGraph.builder.get_join_iterations() == [3, 4, 5, 6, 7]
        print(TestOBMGraph.builder.graph)

    def test_02_build_graph(self):
        TestOBMGraph.builder.build_graph()
        assert compare_graph_to_baseline(TestOBMGraph.builder.graph) is True

    def test_03_visualize_graph(self):
        out = TestOBMGraph.builder.graph.render(view=True)
        assert out == 'Digraph.gv.pdf'
class TestOBMLalePipeline(unittest.TestCase):
    obm_json_path = './autoai/artifacts/agent_perf/obm.json'
    pipeline_json = None
    builder = None

    @classmethod
    def setUp(cls) -> None:
        with open(cls.obm_json_path) as json_file:
            cls.obm_json = json.load(json_file)

        assert len(TestOBMLalePipeline.obm_json) > 0

    def test_01_initialize_graph_object(self):
        TestOBMLalePipeline.builder = OBMPipelineGraphBuilder(TestOBMLalePipeline.obm_json)

        assert isinstance(TestOBMLalePipeline.builder, OBMPipelineGraphBuilder)

    def test_02_build_lale_pipeline(self):
        lale_pipeline_prefix = TestOBMLalePipeline.builder.build_preprocessing_prefix()
        assert lale_pipeline_prefix.name() == 'Relational'
        preprocessing_prefix = lale_pipeline_prefix._impl.operator
        assert preprocessing_prefix is not None
        assert len(preprocessing_prefix.steps()) > 1


if __name__ == '__main__':
    unittest.main()
