#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from tabulate import tabulate
import copy
import logging
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES,PIPELINES, EXPERIMENTS, RUNTIMES, LIBRARIES, DEPLOYMENTS

import json

logger = logging.getLogger(__name__)


class MetaProp:
    def __init__(self, name, key, prop_type, required, example_value, ignored=False, hidden=False, default_value='', schema= '',path=None, transform=lambda x, client: x):
        self.key = key
        self.name = name
        self.prop_type = prop_type
        self.required = required
        self.example_value = example_value
        self.ignored = ignored
        self.hidden = hidden
        self.schema = schema
        self.default_value = default_value
        self.path = path if path is not None else '/' + key
        self.transform = transform


class MetaNamesBase:
    def __init__(self, meta_props_definitions):
        self._meta_props_definitions = meta_props_definitions

    def _validate(self, meta_props):
        for meta_prop in self._meta_props_definitions:
            if meta_prop.key != MetaNames.OUTPUT_DATA_SCHEMA and meta_prop.key != MetaNames.INPUT_DATA_SCHEMA:
                if meta_prop.ignored is False:
                    WMLResource._validate_meta_prop(meta_props, meta_prop.key, meta_prop.prop_type, meta_prop.required)
                else:
                  if(meta_prop.key in meta_props):
                    logger.warning('\'{}\' meta prop is deprecated. It will be ignored.'.format(meta_prop.name))

    def _check_types_only(self, meta_props):
        for meta_prop in self._meta_props_definitions:
            if meta_prop.ignored is False:
                WMLResource._validate_meta_prop(meta_props, meta_prop.key, meta_prop.prop_type, False)
            else:
              if (meta_prop.key in meta_props):
                logger.warning('\'{}\' meta prop is deprecated. It will be ignored.'.format(meta_prop.name))

    def get(self):
        return sorted(list(map(lambda x: x.name, filter(lambda x: not x.ignored and not x.hidden, self._meta_props_definitions))))

    def show(self):
        print(self._generate_table())

    def _generate_doc_table(self):
        return self._generate_table('MetaName', 'Type', 'Required', 'Default value', 'Schema', 'Example value',
                                    show_examples=True, format='grid', values_format='``{}``')

    def _generate_doc(self, resource_name):
        return """
Set of MetaNames for {}.

Available MetaNames:

{}

""".format(resource_name, MetaNamesBase(self._meta_props_definitions)._generate_doc_table())


    def _generate_table(self, name_label='META_PROP NAME', type_label='TYPE',
                       required_label='REQUIRED', default_value_label='DEFAULT_VALUE',
                       example_value_label='EXAMPLE_VALUE', schema_label='SCHEMA',show_examples=False, format='simple', values_format='{}'):

        show_defaults = any(meta_prop.default_value != '' for meta_prop in filter(lambda x: not x.ignored and not x.hidden, self._meta_props_definitions))
        show_schema = any(meta_prop.schema != '' for meta_prop in filter(lambda x: not x.ignored and not x.hidden, self._meta_props_definitions))

        header = [name_label, type_label, required_label]
        if show_schema:
            header.append(schema_label)

        if show_defaults:
            header.append(default_value_label)

        if show_examples:
            header.append(example_value_label)

        table_content = []

        for meta_prop in filter(lambda x: not x.ignored and not x.hidden, self._meta_props_definitions):
            row = [meta_prop.name, meta_prop.prop_type.__name__, u'Y' if meta_prop.required else u'N']

            if show_defaults:
                row.append(values_format.format(meta_prop.default_value) if meta_prop.default_value != '' else '')

            if show_examples:
                row.append(values_format.format(meta_prop.example_value) if meta_prop.example_value != '' else '')

            if show_schema:
                row.append(values_format.format(meta_prop.schema) if meta_prop.schema != '' else '')

            table_content.append(row)

        table = tabulate(
            [header] + table_content,
            tablefmt=format
        )
        return table

    def get_example_values(self):
        return dict((x.key, x.example_value) for x in filter(lambda x: not x.ignored and not x.hidden, self._meta_props_definitions))

    def _generate_resource_metadata(self, meta_props, client=None, with_validation=False, initial_metadata={}):
        if with_validation:
            self._validate(meta_props)

        metadata = copy.deepcopy(initial_metadata)

        def update_map(m, path, el):
            if type(m) is dict:
                if len(path) == 1:
                    m[path[0]] = el
                else:
                    if path[0] not in m:
                        if type(path[1]) is not int:
                            m[path[0]] = {}
                        else:
                            m[path[0]] = []
                    update_map(m[path[0]], path[1:], el)
            elif type(m) is list:
                if len(path) == 1:
                    if len(m) > len(path):
                        m[path[0]] = el
                    else:
                        m.append(el)
                else:
                    if len(m) <= path[0]:
                        m.append({})
                    update_map(m[path[0]], path[1:], el)
            else:
                raise WMLClientError('Unexpected metadata path type: {}'.format(type(m)))


        for meta_prop_def in filter(lambda x: not x.ignored, self._meta_props_definitions):
            if meta_prop_def.key in meta_props:

                path = [int(p) if p.isdigit() else p for p in meta_prop_def.path.split('/')[1:]]

                update_map(
                    metadata,
                    path,
                    meta_prop_def.transform(meta_props[meta_prop_def.key], client)
                )

        return metadata

    def _generate_patch_payload(self,
                                current_metadata,
                                meta_props,
                                client=None,
                                with_validation=False):
        if with_validation:
            self._check_types_only(meta_props)

        def _generate_patch_payload_simple(meta_props_, current_metadata_):
            updated_metadata = self._generate_resource_metadata(meta_props_, client, False, current_metadata_)

            patch_payload = []

            def contained_path(metadata, path):
                if path[0] in metadata:
                    if len(path) == 1:
                        return [path[0]]
                    else:
                        rest_of_path = contained_path(metadata[path[0]], path[1:])
                        if rest_of_path is None:
                            return [path[0]]
                        else:
                            return [path[0]] + rest_of_path
                else:
                    return []

            def get_value(metadata, path):
                if len(path) == 1:
                    return metadata[path[0]]
                else:
                    return get_value(metadata[path[0]], path[1:])

            def already_in_payload(path):
                return any([el['path'] == path for el in patch_payload])

            def update_payload(path):
                existing_path = contained_path(current_metadata_, path)

                if len(existing_path) == len(path):
                    patch_payload.append({
                        'op': 'replace',
                        'path': '/' + '/'.join(existing_path),
                        'value': get_value(updated_metadata, existing_path)
                    })
                else:
                    if not already_in_payload(existing_path):
                            patch_payload.append({
                            'op': 'add',
                            'path': '/' + '/'.join(existing_path + [path[len(existing_path)]]),
                            'value': get_value(updated_metadata, existing_path + [path[len(existing_path)]])
                     })

            for meta_prop_def in filter(lambda x: not x.ignored, self._meta_props_definitions):
                if meta_prop_def.key in meta_props_:

                    path = [int(p) if p.isdigit() else p for p in meta_prop_def.path.split('/')[1:]]

                    update_payload(path)

            return patch_payload

        metadata_props = {m: meta_props[m] for d in self._meta_props_definitions for m in meta_props if d.key == m and d.path.startswith('/metadata/')}
        entity_props = {m: meta_props[m] for d in self._meta_props_definitions for m in meta_props if d.key == m and not d.path.startswith('/metadata/')}

        res = []

        if len(metadata_props) > 0:
            res += _generate_patch_payload_simple(metadata_props, current_metadata)

        if len(entity_props) > 0:
            res += _generate_patch_payload_simple(entity_props, current_metadata['entity'] if 'entity' in current_metadata else current_metadata)

        return res


class TrainingConfigurationMetaNames(MetaNamesBase):
    TAGS = "tags"
    EXPERIMENT_UID = "experiment_uid"
    PIPELINE_UID = "pipeline_uid"
    TRAINING_LIB = "training_lib"
    TRAINING_LIB_UID = "training_lib_uid"
    TRAINING_LIB_MODEL_TYPE = "model_type"
    TRAINING_LIB_RUNTIME_UID = "runtime"
    TRAINING_LIB_PARAMETERS = "parameters"
    COMMAND = "command"
    COMPUTE = "compute"
    PIPELINE_DATA_BINDINGS = "data_bindings"
    PIPELINE_NODE_PARAMETERS = "node_parameters"
    PIPELINE_MODEL_TYPE = "model_type"
    SPACE_UID = "space_uid"

    TRAINING_DATA_REFERENCES = "training_data_references"
    TEST_DATA_REFERENCES = "test_data_references"
    TRAINING_RESULTS_REFERENCE = "results_reference"
    _COMPUTE_CONFIGURATION_DEFAULT = u'k80'
    _meta_props_definitions = [
        MetaProp('TRAINING_DATA_REFERENCES',    TRAINING_DATA_REFERENCES,              list,       True,   [{u'connection': {u'href': u'/v2/connections/2d07a6b4-8fa9-43ab-91c8-befcd9dab8d2?space_id=440ada9b-af87-4da8-a9fa-a5450825e260'},u'location': {u'bucket': u'train-data',u'path':u'training_path'},u'type': u'connection_asset',u'schema':{u'id':u'1', u'fields': [{u'name': u'x', u'type': u'double', u'nullable': u'False'}]}}] ,schema=[{u'name(optional)': u'string',u'type(required)': u'string',u'connection(required)':{u'href(required)': u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'},u'schema(optional)':{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}}]),
        MetaProp('TRAINING_RESULTS_REFERENCE',  TRAINING_RESULTS_REFERENCE,            dict,       True,   {u'connection': {u'href': u'/v2/connections/2d07a6b4-8fa9-43ab-91c8-befcd9dab8d2?space_id=440ada9b-af87-4da8-a9fa-a5450825e260'},u'location': {u'bucket': u'test-results',u'path':u'training_path'},u'type': u'connection_asset'},schema={u'name(optional)': u'string',u'type(required)': u'string',u'connection(required)':{u'href(required)': u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'}}),
        MetaProp('TEST_DATA_REFERENCES',         TEST_DATA_REFERENCES,                 list,       False,   [{u'connection': {u'href': u'/v2/connections/2d07a6b4-8fa9-43ab-91c8-befcd9dab8d2?space_id=440ada9b-af87-4da8-a9fa-a5450825e260'},u'location': {u'bucket': u'train-data',u'path':u'training_path'},u'type': u'connection_asset',u'schema':{u'id':u'1', u'fields': [{u'name': u'x', u'type': u'double', u'nullable': u'False'}]}}] ,schema=[{u'name(optional)': u'string',u'type(required)': u'string',u'connection(required)':{u'href(required)': u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'},u'schema(optional)':{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}}]),
        MetaProp('TAGS',                        TAGS                                 , list,       False,  [{u'value': u"string",u"description": u"string"}],schema=[{u'value(required)': u'string',u'description(optional)': u'string'}]),
        MetaProp('PIPELINE_UID',                PIPELINE_UID,                          str,   False, example_value="3c1ce536-20dc-426e-aac7-7284cf3befc6", path="/pipeline/href",transform=lambda x, client: API_VERSION+PIPELINES+"/"+x),
        MetaProp('EXPERIMENT_UID',              EXPERIMENT_UID,                        str,       False, example_value="3c1ce536-20dc-426e-aac7-7284cf3befc6", path="/experiment/href",transform=lambda x, client: API_VERSION + EXPERIMENTS + "/" + x),
        MetaProp('PIPELINE_DATA_BINDINGS',      PIPELINE_DATA_BINDINGS,                str,  False,  path="/pipeline/data_bindings",example_value=[{ "data_reference_name": "string","node_id": "string"}],schema=[{"data_reference_name(required)":"string","node_id(required)":"string"}]),
        MetaProp('PIPELINE_NODE_PARAMETERS',    PIPELINE_NODE_PARAMETERS,              dict,       False,  path="/pipeline/node_parameters",example_value=[{"node_id": "string","parameters": {}}],schema=[{"node_id(required)":"string","parameters(required)":"dict"}] ),
        MetaProp('SPACE_UID', SPACE_UID, str, False, u'3c1ce536-20dc-426e-aac7-7284cf3befc6', path='/space/href',
                 transform=lambda x, client: API_VERSION + SPACES + "/" + x),
        MetaProp('TRAINING_LIB',                TRAINING_LIB,                          dict,       False, example_value={"href": "/v4/libraries/3c1ce536-20dc-426e-aac7-7284cf3befc6","compute": {"name": "k80","nodes": 0},"runtime": { "href": "/v4/runtimes/3c1ce536-20dc-426e-aac7-7284cf3befc6"},"command": "python3 convolutional_network.py","parameters": {}}, path="/training_lib",schema={"href(required)":"string","type(required)":"string","runtime(optional)":{"href":"string"},"command(optional)":"string","parameters(optional)":"dict"}),
        MetaProp('TRAINING_LIB_UID', TRAINING_LIB_UID, str, False, example_value="3c1ce536-20dc-426e-aac7-7284cf3befc6",
                 path="/training_lib/href"),
        MetaProp('TRAINING_LIB_MODEL_TYPE', TRAINING_LIB_MODEL_TYPE, str, False, example_value="3c1ce536-20dc-426e-aac7-7284cf3befc6",
                 ),
        MetaProp('TRAINING_LIB_RUNTIME_UID', TRAINING_LIB_RUNTIME_UID, str, False, example_value="3c1ce536-20dc-426e-aac7-7284cf3befc6",
                 ),
        MetaProp('TRAINING_LIB_PARAMETERS', TRAINING_LIB_PARAMETERS, dict, False, example_value="3c1ce536-20dc-426e-aac7-7284cf3befc6",
                 ),
        MetaProp('COMMAND', COMMAND, str, False,
                 example_value="3c1ce536-20dc-426e-aac7-7284cf3befc6",
                 ),
        MetaProp('COMPUTE', COMPUTE, dict, False,
                 example_value="3c1ce536-20dc-426e-aac7-7284cf3befc6",
                 ),
        MetaProp('PIPELINE_MODEL_TYPE',         PIPELINE_MODEL_TYPE,                   str,        False, example_value="tensorflow_1.1.3-py3",path="/pipeline/model_type")
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('trainings')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class TrainingConfigurationMetaNamesCp4d30(MetaNamesBase):
    TAGS = "tags"
    EXPERIMENT = "experiment"
    PIPELINE = "pipeline"
    MODEL_DEFINITION = "model_definition"
    FEDERATED_LEARNING = "federated_learning"
    NAME = "name"
    DESCRIPTION = "description"
    SPACE_UID = "space_uid"
    TRAINING_DATA_REFERENCES = "training_data_references"
    TEST_DATA_REFERENCES = "test_data_references"
    TEST_OUTPUT_DATA = "test_output_data"
    TRAINING_RESULTS_REFERENCE = "results_reference"
    _COMPUTE_CONFIGURATION_DEFAULT = u'k80'
    _meta_props_definitions = [
        MetaProp('TRAINING_DATA_REFERENCES',     TRAINING_DATA_REFERENCES,             list,       True,   [{u'connection': {u'endpoint_url': u'https://s3-api.us-geo.objectstorage.softlayer.net',u'access_key_id': u'***',u'secret_access_key': u'***'},u'location': {u'bucket': u'train-data',u'path':u'training_path'},u'type': u's3',u'schema':{u'id':u'1', u'fields': [{u'name': u'x', u'type': u'double', u'nullable': u'False'}]}}] ,schema=[{u'name(optional)': u'string',u'type(required)': u'string',u'connection(required)':{u'endpoint_url(required)': u'string',u'access_key_id(required)':u'string',u'secret_access_key(required)':u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'},u'schema(optional)':{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}}]),
        MetaProp('TRAINING_RESULTS_REFERENCE',   TRAINING_RESULTS_REFERENCE,            dict,       True,   {u'connection': {u'endpoint_url': u'https://s3-api.us-geo.objectstorage.softlayer.net',u'access_key_id': u'***',u'secret_access_key': u'***'},u'location': {u'bucket': u'test-results',u'path':u'training_path'},u'type': u's3'},schema={u'name(optional)': u'string',u'type(required)': u'string',u'connection(required)':{u'endpoint_url(required)': u'string',u'access_key_id(required)':u'string',u'secret_access_key(required)':u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'}}),
        MetaProp('TEST_DATA_REFERENCES',         TEST_DATA_REFERENCES,                  list,       False,   [{u'connection': {u'endpoint_url': u'https://s3-api.us-geo.objectstorage.softlayer.net',u'access_key_id': u'***',u'secret_access_key': u'***'},u'location': {u'bucket': u'train-data',u'path':u'training_path'},u'type': u's3',u'schema':{u'id':u'1', u'fields': [{u'name': u'x', u'type': u'double', u'nullable': u'False'}]}}] ,schema=[{u'name(optional)': u'string',u'type(required)': u'string',u'connection(required)':{u'endpoint_url(required)': u'string',u'access_key_id(required)':u'string',u'secret_access_key(required)':u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'},u'schema(optional)':{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}}]),
        MetaProp('TEST_OUTPUT_DATA',             TEST_OUTPUT_DATA,                      dict,       False, [{u'connection': {u'endpoint_url': u'https://s3-api.us-geo.objectstorage.softlayer.net', u'access_key_id': u'***',u'secret_access_key': u'***'}, u'location': {u'bucket': u'train-data', u'path': u'training_path'},u'type': u's3', u'schema': {u'id': u'1',u'fields': [{u'name': u'x',u'type': u'double',u'nullable': u'False'}]}}],schema={u'name(optional)': u'string', u'type(required)': u'string',u'connection(required)': {u'endpoint_url(required)': u'string',u'access_key_id(required)': u'string',u'secret_access_key(required)': u'string'},u'location(required)': {u'bucket': u'string', u'path': u'string'},u'schema(optional)': {u'id(required)': u'string', u'fields(required)': [{u'name(required)': u'string', u'type(required)': u'string',u'nullable(optional)': u'string'}]}}),
        MetaProp('TAGS',                        TAGS                                 , list,       False,  [u"string"],schema=[u'string']),
        MetaProp('PIPELINE',                PIPELINE,                          dict,   False, {"id":"3c1ce536-20dc-426e-aac7-7284cf3befc6","rev":"1","modeltype":"tensorflow_1.1.3-py3", "data_bindings":[{ "data_reference_name":"string","node_id": "string"}],"node_parameters":[{"node_id": "string","parameters": {}}],"hardware_spec":{"id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab","rev": "12","name": "string","num_nodes": "2"}, "hybrid_pipeline_hardware_specs": [{"node_runtime_id": "string","hardware_spec": {"id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab","rev": "12","name": "string", "num_nodes": "2"}}]}),
        MetaProp('EXPERIMENT',              EXPERIMENT,                        dict,       False, {"id":"3c1ce536-20dc-426e-aac7-7284cf3befc6", "rev": 1, "description": "test experiment"}),
        MetaProp('FEDERATED_LEARNING', FEDERATED_LEARNING, dict, False, example_value="3c1ce536-20dc-426e-aac7-7284cf3befc6",
                 path="/federated_learning"),

        MetaProp('SPACE_UID', SPACE_UID, str, False, u'3c1ce536-20dc-426e-aac7-7284cf3befc6', path='/space/href',
                 transform=lambda x, client: x),
        MetaProp('MODEL_DEFINITION',                MODEL_DEFINITION,                          dict,       False, {"id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab","rev": "12", "model_type": "string","hardware_spec": {"id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab","rev": "12","name": "string","num_nodes": "2"},"software_spec": {"id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab","rev": "12", "name": "..."},"command": "string","parameters": {}}),
        MetaProp('DESCRIPTION', DESCRIPTION, str, True, example_value="tensorflow model training"),
        MetaProp('NAME',        NAME,                   str,        True, example_value="sample training")
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('trainings')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class ExperimentMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    TAGS = "tags"
    EVALUATION_METHOD = "evaluation_method"
    EVALUATION_METRICS = "evaluation_metrics"
    TRAINING_REFERENCES = "training_references"
    SPACE_UID = "space_uid"
    LABEL_COLUMN = "label_column"
    CUSTOM = "custom"




    _meta_props_definitions = [
        MetaProp('NAME',                            NAME,                           str,        True,     u'Hand-written Digit Recognition'),
        MetaProp('DESCRIPTION',                     DESCRIPTION,                    str,        False,    u'Hand-written Digit Recognition training'),
        MetaProp('TAGS',                            TAGS,                           list,       False,    [{u'value': 'dsx-project.<project-guid>',u'description': 'DSX project guid'}],schema=[{u'value(required)': u'string',u'description(optional)': u'string'}]),
        MetaProp('EVALUATION_METHOD',               EVALUATION_METHOD,              str,        False,    u'multiclass', path="/evaluation_definition/method"),
        MetaProp('EVALUATION_METRICS',              EVALUATION_METRICS,             list,       False,    [{u'name':u'accuracy', u'maximize': False}], path="/evaluation_definition/metrics", schema=[{u'name(required)': u'string', u'maximize(optional)': u'boolean'}]),
        MetaProp('TRAINING_REFERENCES',             TRAINING_REFERENCES,            list,       True,     [{u'pipeline': {u'href': u'/v4/pipelines/6d758251-bb01-4aa5-a7a3-72339e2ff4d8'}}],schema=[{u'pipeline(optional)': {u'href(required)': u'string', u'data_bindings(optional)':[{u'data_reference(required)':u'string', u'node_id(required)':u'string'}], u'nodes_parameters(optional)':[{u'node_id(required)':u'string', u'parameters(required)':u'dict'}]}, u'training_lib(optional)': {u'href(required)': u'string', u'compute(optional)': {u'name(required)': u'string', u'nodes(optional)': u'number'}, u'runtime(optional)': {u'href(required)': u'string'}, u'command(optional)': u'string', u'parameters(optional)':u'dict'}}]),
        MetaProp('SPACE_UID',                       SPACE_UID,                      str,        False,    u'3c1ce536-20dc-426e-aac7-7284cf3befc6', path='/space/href', transform=lambda x, client: API_VERSION + SPACES + "/" + x),
        MetaProp('LABEL_COLUMN',                    LABEL_COLUMN,                   str ,       False,    u'label'),
        MetaProp('CUSTOM',                          CUSTOM,                         dict,       False,     {"field1": "value1"})

    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('experiments')


    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class PipelineMetanames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    TAGS = "tags"
    SPACE_UID = "space_url"
    IMPORT = "import"
    DOCUMENT = "document"
    CUSTOM = "custom"
    RUNTIMES = "runtimes"
    COMMAND = "command"
    LIBRARY_UID = "training_lib_href"
    COMPUTE = "compute"

    _meta_props_definitions = [
        MetaProp('NAME',                        NAME,                         str,        True,     u'Hand-written Digit Recognitionu'),
        MetaProp('DESCRIPTION',                 DESCRIPTION,                  str,        False,    u'Hand-written Digit Recognition training'),
        MetaProp('SPACE_UID',                   SPACE_UID,                    str,        False,    u'3c1ce536-20dc-426e-aac7-7284cf3befc6',path="/space/href",transform=lambda x, client: API_VERSION+SPACES+"/"+x),
        MetaProp('TAGS',                        TAGS,                         list,       False,    [{u'value': 'dsx-project.<project-guid>',u'description': 'DSX project guid'}],schema=[{u'value(required)': u'string',u'description(optional)': u'string'}]),
        MetaProp('DOCUMENT',                    DOCUMENT,                     dict,       False,    example_value={"doc_type":"pipeline","version": "2.0","primary_pipeline": "dlaas_only","pipelines": [{"id": "dlaas_only","runtime_ref": "hybrid","nodes": [{"id": "training","type": "model_node","op": "dl_train","runtime_ref": "DL","inputs": [],"outputs": [],"parameters": {"name": "tf-mnist","description": "Simple MNIST model implemented in TF","command": "python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000","compute": {"name": "k80","nodes": 1},"training_lib_href":"/v4/libraries/64758251-bt01-4aa5-a7ay-72639e2ff4d2/content"},"target_bucket": "wml-dev-results"}]}]},schema={"doc_type(required)":"string","version(required)": "string","primary_pipeline(required)": "string","pipelines(required)": [{"id(required)": "string","runtime_ref(required)": "string","nodes(required)": [{"id": "string","type": "string","inputs": "list","outputs": "list","parameters": {"training_lib_href":"string"}}]}]}),
        MetaProp('CUSTOM',                      CUSTOM,                       dict,       False,    example_value={"field1":"value1"}),
        MetaProp('IMPORT',                      IMPORT,                       dict,       False,    example_value={u'connection': {u'endpoint_url': u'https://s3-api.us-geo.objectstorage.softlayer.net',u'access_key_id': u'***',u'secret_access_key': u'***'},u'location': {u'bucket': u'train-data',u'path':u'training_path'},u'type': u's3'},schema={u'name(optional)': u'string',u'type(required)': u'string',u'connection(required)':{u'endpoint_url(required)': u'string',u'access_key_id(required)':u'string',u'secret_access_key(required)':u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'}}),
        MetaProp('RUNTIMES',                    RUNTIMES,                     list,       False,    example_value=[{"id":"id","name":"tensorflow","version":"1.13-py3"}]),
        MetaProp('COMMAND',                     COMMAND,                      str,        False,    example_value="convolutional_network.py --trainImagesFile train-images-idx3-ubyte.gz --trainLabelsFile train-labels-idx1-ubyte.gz --testImagesFile t10k-images-idx3-ubyte.gz --testLabelsFile t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000"),
        MetaProp("LIBRARY_UID",                 LIBRARY_UID,                  str,        False,    example_value="fb9752c9-301a-415d-814f-cf658d7b856a"),
        MetaProp("COMPUTE",                     COMPUTE,                      dict,       False,    example_value={"name":"k80","nodes":1})
        ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('pipelines')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class LearningSystemMetaNames(MetaNamesBase):
    _COMPUTE_CONFIGURATION_DEFAULT = 'k80'
    FEEDBACK_DATA_REFERENCE = "feedback_data_reference"
    SPARK_REFERENCE = "spark_instance"
    MIN_FEEDBACK_DATA_SIZE = "min_feedback_data_size"
    AUTO_RETRAIN = "auto_retrain"
    AUTO_REDEPLOY = "auto_redeploy"
    COMPUTE_CONFIGURATION = "compute_configuration"
    TRAINING_RESULTS_REFERENCE = "training_results_reference"

    _meta_props_definitions = [
        MetaProp('FEEDBACK_DATA_REFERENCE', FEEDBACK_DATA_REFERENCE,     dict,       True, example_value={}),
        MetaProp('SPARK_REFERENCE',         SPARK_REFERENCE,             dict,       False, example_value={}),
        MetaProp('MIN_FEEDBACK_DATA_SIZE',  MIN_FEEDBACK_DATA_SIZE,      int,        True, example_value=100),
        MetaProp('AUTO_RETRAIN',            AUTO_RETRAIN,                str,        True, example_value="conditionally"),
        MetaProp('AUTO_REDEPLOY',           AUTO_REDEPLOY,               str,        True, example_value="always"),
        MetaProp('COMPUTE_CONFIGURATION',   COMPUTE_CONFIGURATION,       dict,       False, example_value={u'name': _COMPUTE_CONFIGURATION_DEFAULT}),
        MetaProp('TRAINING_RESULTS_REFERENCE', TRAINING_RESULTS_REFERENCE, dict,        False, example_value={u'connection': {u'endpoint_url': u'https://s3-api.us-geo.objectstorage.softlayer.net', u'access_key_id': u'***', u'secret_access_key': u'***'},u'target': {u'bucket': u'train-data'}, u'type': u's3'}),
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('learning system')

class MemberMetaNames(MetaNamesBase):
    IDENTITY = "identity"
    ROLE = "role"
    IDENTITY_TYPE = "identity_type"

    _meta_props_definitions = [
        MetaProp('IDENTITY',                        IDENTITY,               str, True,   "IBMid-060000123A (service-ID or IAM-userID)"),
        MetaProp('ROLE', ROLE, str, True, "Supported values - Viewer/Editor/Admin"),
        MetaProp('IDENTITY_USER', IDENTITY_TYPE, str, True, "Supported values - service/user")
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Member Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class ModelMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = MetaNames.DESCRIPTION
    TRAINING_DATA_REFERENCES = "training_data_references"
    TEST_DATA_REFERENCES = "test_data_references"
    OUTPUT_DATA_SCHEMA = MetaNames.OUTPUT_DATA_SCHEMA
    LABEL_FIELD = MetaNames.LABEL_FIELD
    TRANSFORMED_LABEL_FIELD = MetaNames.TRANSFORMED_LABEL_FIELD
    RUNTIME_UID = "runtime"
    INPUT_DATA_SCHEMA = MetaNames.INPUT_DATA_SCHEMA
    CUSTOM="custom"
    DOMAIN="domain"
    HYPER_PARAMETERS="hyper_parameters"
    TAGS = "tags"
    SPACE_UID = "space"
    PIPELINE_UID = "pipeline"
    TYPE="type"
    SIZE="size"
    IMPORT = "import"
    TRAINING_LIB_UID = "training_lib"
    MODEL_DEFINITION_UID = "model_definition"
    METRICS="metrics"
    SOFTWARE_SPEC_UID = "software_spec"
    TF_MODEL_PARAMS = "tf_model_params"
    FAIRNESS_INFO = "fairness_info"

    _meta_props_definitions = [
        MetaProp('NAME',                    NAME,                        str,        True,   "my_model"),
        MetaProp('DESCRIPTION',             DESCRIPTION,                 str,        False,  "my_description", path="/description"),
        MetaProp('INPUT_DATA_SCHEMA',       INPUT_DATA_SCHEMA,           list,       False,  example_value={"id":"1","type": "struct", "fields": [{"name": "x", "type": "double", "nullable": False, "metadata": {}}, {"name": "y", "type": "double", "nullable": False, "metadata": {}}]}, path="/input_data_schema",schema={u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}),
        MetaProp('TRAINING_DATA_REFERENCES',TRAINING_DATA_REFERENCES,    list,       False,  [], path="/training_data_references",schema=[{"name(optional)":"string","type(required)":"string",u'connection(required)':{u'endpoint_url(required)': u'string',u'access_key_id(required)':u'string',u'secret_access_key(required)':u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'},u'schema(optional)':{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}}]),
        MetaProp('TEST_DATA_REFERENCES',    TEST_DATA_REFERENCES,        list,       False,  [], path="/test_data_references", schema=[{"name(optional)": "string", "type(required)": "string", u'connection(required)': {u'endpoint_url(required)': u'string', u'access_key_id(required)': u'string', u'secret_access_key(required)': u'string'}, u'location(required)': {u'bucket': u'string', u'path': u'string'}, u'schema(optional)': {u'id(required)': u'string', u'fields(required)': [{u'name(required)': u'string', u'type(required)': u'string', u'nullable(optional)': u'string'}]}}]),
        MetaProp('OUTPUT_DATA_SCHEMA',      OUTPUT_DATA_SCHEMA,          dict,       False,  example_value={"id":"1","type": "struct", "fields": [{"name": "x", "type": "double", "nullable": False, "metadata": {}}, {"name": "y", "type": "double", "nullable": False, "metadata": {}}]}, path="/output_data_schema",schema={u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}),
        MetaProp('LABEL_FIELD',             LABEL_FIELD,                 str,        False,  example_value='PRODUCT_LINE', path="/label_column"),
        MetaProp('TRANSFORMED_LABEL_FIELD', TRANSFORMED_LABEL_FIELD,     str,        False,  example_value='PRODUCT_LINE_IX', path="/transformed_label"),
        MetaProp('TAGS',                    TAGS,                        list,       False,  example_value=["string","string"],schema=[ u'string', u'string']),
        MetaProp('SIZE',                    SIZE,                        dict,       False, example_value={"in_memory": 0,"content": 0},schema={u'in_memory(optional)': u'string', u'content(optional)': u'string'}),
        MetaProp('SPACE_UID',               SPACE_UID,                   str,        False,  example_value="53628d69-ced9-4f43-a8cd-9954344039a8", path="/space/href", hidden=True),
        MetaProp('PIPELINE_UID',            PIPELINE_UID,                str,        False,  example_value="53628d69-ced9-4f43-a8cd-9954344039a8", path="/pipeline/href"),
        MetaProp('RUNTIME_UID',             RUNTIME_UID,                 str,        False, example_value="53628d69-ced9-4f43-a8cd-9954344039a8",path="/runtime/href"),
        MetaProp('TYPE',                    TYPE,                        str,        True,  example_value="mllib_2.1", path="/type"),
        MetaProp('CUSTOM',                  CUSTOM ,                     dict,       False, example_value={}),
        MetaProp('DOMAIN',                  DOMAIN,                      str,        False, example_value="Watson Machine Learning"),
        MetaProp('HYPER_PARAMETERS',        HYPER_PARAMETERS,            dict,       False, example_value=""),
        MetaProp('METRICS',                 METRICS,                     list,       False, example_value=""),
        MetaProp('IMPORT',                  IMPORT,                      dict,       False,  example_value={u'connection': {u'endpoint_url': u'https://s3-api.us-geo.objectstorage.softlayer.net',u'access_key_id': u'***',u'secret_access_key': u'***'},u'location': {u'bucket': u'train-data',u'path':u'training_path'},u'type': u's3'},schema={u'name(optional)': u'string',u'type(required)': u'string',u'connection(required)':{u'endpoint_url(required)': u'string',u'access_key_id(required)':u'string',u'secret_access_key(required)':u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'}}),
        MetaProp('TRAINING_LIB_UID',        TRAINING_LIB_UID,            str,        False,  example_value="53628d69-ced9-4f43-a8cd-9954344039a8", path="/training_lib"),
        MetaProp('MODEL_DEFINITION_UID',    MODEL_DEFINITION_UID,        str,        False, example_value="53628d6_cdee13-35d3-s8989343", path="/model_definition"),
        MetaProp('SOFTWARE_SPEC_UID',       SOFTWARE_SPEC_UID,           str,        False, example_value="53628d69-ced9-4f43-a8cd-9954344039a8", path="/software_spec/id", transform=lambda x, client: x),
        MetaProp('TF_MODEL_PARAMS',         TF_MODEL_PARAMS,             dict,       False, example_value={"save_format": "None", "signatures": "struct","options": "None", "custom_objects": "string"},path="/tf_model_params"),
        MetaProp('FAIRNESS_INFO',           FAIRNESS_INFO,               dict,       False, example_value={"favorable_labels": ["X"]}, path="/metrics/0/context/fairness/info")

    ]


    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('models') + '''
**Note:** `project` (MetaNames.PROJECT_UID) and `space` (MetaNames.SPACE_UID) meta names are not supported and considered as invalid. Instead use client.set.default_space(<SPACE_GUID>) to set the space or client.set.default_project(<PROJECT_GUID>).
    '''

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class PayloadLoggingMetaNames(MetaNamesBase):
    PAYLOAD_DATA_REFERENCE = "payload_store"
    LABELS = "labels"
    OUTPUT_DATA_SCHEMA = "output_data_schema"

    _meta_props_definitions = [
        MetaProp('PAYLOAD_DATA_REFERENCE',  PAYLOAD_DATA_REFERENCE,       dict, True,     {}),
        MetaProp('LABELS',                  LABELS,              list, False,    ['a', 'b', 'c']),
        MetaProp('OUTPUT_DATA_SCHEMA',      OUTPUT_DATA_SCHEMA,  dict, False,    {})
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('payload logging system')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class FunctionMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    RUNTIME_UID = "runtime_uid"
    INPUT_DATA_SCHEMAS = "input_data_schemas"
    OUTPUT_DATA_SCHEMAS = "output_data_schemas"
    TAGS = "tags"
    SPACE_UID = "space_uid"
    SOFTWARE_SPEC_UID = "software_spec_uid"
    TYPE = "type"
    CUSTOM="custom"
    SAMPLE_SCORING_INPUT="sample_scoring_input"

    _meta_props_definitions = [
        MetaProp('NAME',                NAME,                str,        True,   "ai_function"),
        MetaProp('DESCRIPTION',         DESCRIPTION,         str,        False,  "This is ai function"),
        MetaProp('RUNTIME_UID',         RUNTIME_UID,         str,        False,  '53628d69-ced9-4f43-a8cd-9954344039a8', path="/runtime/href", transform=lambda x, client: API_VERSION+RUNTIMES+"/"+x),
        MetaProp('SOFTWARE_SPEC_UID',   SOFTWARE_SPEC_UID,   str,        False,  '53628d69-ced9-4f43-a8cd-9954344039a8',
                 path="/software_spec/id", transform=lambda x, client: x),
        MetaProp('INPUT_DATA_SCHEMAS',  INPUT_DATA_SCHEMAS,  list,       False,  [{"id":"1","type": "struct", "fields": [{"name": "x", "type": "double", "nullable": False, "metadata": {}}, {"name": "y", "type": "double", "nullable": False, "metadata": {}}]}],schema=[{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}]),
        MetaProp('OUTPUT_DATA_SCHEMAS', OUTPUT_DATA_SCHEMAS, list,       False,  [{"id":"1","type": "struct", "fields": [{"name": "multiplication", "type": "double", "nullable": False, "metadata": {}}]}],schema=[{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}]),
        MetaProp('TAGS',                TAGS,                list,       False,  [{"value": "ProjectA", "description": "Functions created for ProjectA"}],schema=[{u'value(required)': u'string',u'description(optional)': u'string'}]),
        MetaProp('TYPE',                TYPE,                str,        False,  u'python'),
        MetaProp('CUSTOM',              CUSTOM,              dict,       False,  example_value=u'{}'),
        MetaProp('SAMPLE_SCORING_INPUT', SAMPLE_SCORING_INPUT, list,     False,  example_value={ "input_data": [ { "fields": [ "name",  "age", "occupation" ], "values": [ [ "john",23, "student"], [ "paul", 33, "engineer" ]] }]},schema={"id(optional)":"string","fields(optional)":"array","values(optional)":"array"}),
        MetaProp('SPACE_UID',           SPACE_UID,           str,        False,  u'3628d69-ced9-4f43-a8cd-9954344039a8',path="/space/href",transform=lambda x, client: API_VERSION+SPACES+"/"+x)
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('AI functions')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class FunctionNewMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    INPUT_DATA_SCHEMAS = "input_data_schemas"
    OUTPUT_DATA_SCHEMAS = "output_data_schemas"
    TAGS = "tags"
    SOFTWARE_SPEC_ID = "software_spec_id"
    SOFTWARE_SPEC_UID = "software_spec_id"
    TYPE = "type"
    CUSTOM = "custom"
    SAMPLE_SCORING_INPUT = "sample_scoring_input"

    _meta_props_definitions = [
        MetaProp('NAME',                NAME,                str,   True,   "ai_function"),
        MetaProp('DESCRIPTION',         DESCRIPTION,         str,   False,  "This is ai function"),
        MetaProp('SOFTWARE_SPEC_ID', SOFTWARE_SPEC_ID, str, False, '53628d69-ced9-4f43-a8cd-9954344039a8',
                 path="/software_spec/id", transform=lambda x, client: x),
        MetaProp('SOFTWARE_SPEC_UID', SOFTWARE_SPEC_UID, str, False, '53628d69-ced9-4f43-a8cd-9954344039a8',
                 path="/software_spec/id", transform=lambda x, client: x),
        MetaProp('INPUT_DATA_SCHEMAS',  INPUT_DATA_SCHEMAS,  list,       False,
                 [{"id":"1","type": "struct", "fields": [{"name": "x", "type": "double", "nullable": False, "metadata": {}},
                                                         {"name": "y", "type": "double", "nullable": False, "metadata": {}}]}],
                 schema=[{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}],
                 path="/schemas/input"),
        MetaProp('OUTPUT_DATA_SCHEMAS', OUTPUT_DATA_SCHEMAS, list,       False,
                 [{"id":"1","type": "struct", "fields": [{"name": "multiplication", "type": "double", "nullable": False, "metadata": {}}]}],
                 schema=[{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}],
                 path="/schemas/output"),
        MetaProp('TAGS',                TAGS,                list,       False,  ["tags1", "tags2"], schema=[u'string']),
        MetaProp('TYPE',                TYPE,                str,   False,  u'python'),
        MetaProp('CUSTOM',              CUSTOM,              dict,       False,  example_value=u'{}'),
        MetaProp('SAMPLE_SCORING_INPUT', SAMPLE_SCORING_INPUT, dict,     False,  example_value={"input_data": [ { "fields": [ "name",  "age", "occupation" ], "values": [ [ "john",23, "student"], [ "paul", 33, "engineer" ]] }]},schema={"id(optional)":"string","fields(optional)":"array","values(optional)":"array"})
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('AI functions')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class ScoringMetaNames(MetaNamesBase):
    INPUT_DATA = "input_data"
    INPUT_DATA_REFERENCES = "input_data_references"
    OUTPUT_DATA_REFERENCE = "output_data_reference"
    EVALUATIONS_SPEC = "evaluations_spec"
    ENVIRONMENT_VARIABLES = "environment_variables"
    NAME = "name"

    _meta_props_definitions = [
        MetaProp('NAME',      NAME,        str,   False,   "jobs test"),
        MetaProp('INPUT_DATA',INPUT_DATA,list,False,  path="/scoring/input_data",example_value=[{"fields": ["name","age","occupation"],"values": [["john",23,"student"]]}], schema=[{"name(optional)":"string","id(optional)":"string","fields(optional)":"array[string]","values":"array[array[string]]"}]),
        MetaProp('INPUT_DATA_REFERENCES',         INPUT_DATA_REFERENCES,         list,   False,example_value = "",path="/scoring/input_data_references",schema=[{"id(optional)":"string","type(required)":"string",u'connection(required)':{u'href(required)': u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'},u'schema(optional)':{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}}]),
        MetaProp('OUTPUT_DATA_REFERENCE',         OUTPUT_DATA_REFERENCE,         dict,   False, example_value = "",path="/scoring/output_data_reference", schema={"type(required)":"string",u'connection(required)':{u'href(required)': u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'},u'schema(optional)':{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}}),
        MetaProp('EVALUATIONS_SPEC',  EVALUATIONS_SPEC,  list,       False,  path="/scoring/evaluations",example_value=[{"id": "string","input_target": "string","metrics_names": ["auroc","accuracy"]}], schema=[{"id(optional)":"string", "input_target(optional)":"string","metrics_names(optional)":"array[string]"}]),
        MetaProp('ENVIRONMENT_VARIABLES', ENVIRONMENT_VARIABLES, dict, False, path="/scoring/environment_variables", example_value = {"my_env_var1": "env_var_value1","my_env_var2": "env_var_value2"})
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Scoring')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class DecisionOptimizationMetaNames(MetaNamesBase):
    INPUT_DATA = "input_data"
    INPUT_DATA_REFERENCES = "input_data_references"
    OUTPUT_DATA = "output_data"
    OUTPUT_DATA_REFERENCES = "output_data_references"
    SOLVE_PARAMETERS = "solve_parameters"

    _meta_props_definitions = [
        MetaProp('INPUT_DATA', INPUT_DATA,     list,       False,path="/decision_optimization/input_data",  example_value=[{"fields": ["name","age","occupation"],"values": [["john",23,"student"]]}], schema=[{"name(optional)":"string","id(optional)":"string","fields(optional)":"array[string]","values":"array[array[string]]"}]),
        MetaProp('INPUT_DATA_REFERENCES', INPUT_DATA_REFERENCES,                list,       False,path="/decision_optimization/input_data_references",  example_value=[{"fields": ["name","age","occupation"],"values": [["john",23,"student"]]}], schema=[{"name(optional)":"string","id(optional)":"string","fields(optional)":"array[string]","values":"array[array[string]]"}]),
        MetaProp('OUTPUT_DATA', OUTPUT_DATA,                list,   False,example_value="",path="/decision_optimization/output_data",  schema=[{"name(optional)": "string"}]),
        MetaProp('OUTPUT_DATA_REFERENCES', OUTPUT_DATA_REFERENCES, list, False,example_value="",path="/decision_optimization/output_data_references",  schema={"name(optional)":"string","type(required)":"string",u'connection(required)':{u'endpoint_url(required)': u'string',u'access_key_id(required)':u'string',u'secret_access_key(required)':u'string'},u'location(required)':{u'bucket':u'string',u'path':u'string'},u'schema(optional)':{u'id(required)':u'string',u'fields(required)':[{u'name(required)':u'string',u'type(required)':u'string',u'nullable(optional)':u'string'}]}}),
        MetaProp('SOLVE_PARAMETERS', SOLVE_PARAMETERS, dict,     False, example_value="",path="/decision_optimization/solve_parameters")
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Decision Optimization')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class RuntimeMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    CUSTOM = "custom"
    PLATFORM = "platform"
    LIBRARIES_UIDS = "libraries_uids"
    CONFIGURATION_FILEPATH = "configuration_filepath"
    TAGS = "tags"
    SPACE_UID = "space"
    COMPUTE = "compute"

    _meta_props_definitions = [
        MetaProp('NAME',                            NAME,                            str,        True,   "runtime_spec_python_3.5"),
        MetaProp('DESCRIPTION',                     DESCRIPTION,                     str,        False,  "sample runtime"),
        MetaProp('PLATFORM',                        PLATFORM,                        dict,       True,   u'{"name":python","version":"3.5")',schema={"name(required)":"string","version(required)":"version"}),
        MetaProp('LIBRARIES_UIDS',                  LIBRARIES_UIDS,                  list,       False,   ["46dc9cf1-252f-424b-b52d-5cdd9814987f"]),
        MetaProp('CONFIGURATION_FILEPATH',          CONFIGURATION_FILEPATH,          str,        False,   "/home/env_config.yaml"),
        MetaProp('TAGS',                            TAGS,                            list,       False,   [{u'value': 'dsx-project.<project-guid>',u'description': 'DSX project guid'}],schema=[{u'value(required)': u'string',u'description(optional)': u'string'}]),
        MetaProp('CUSTOM',                          CUSTOM,                          dict,       False,   u'{"field1": "value1"}'),
        MetaProp('SPACE_UID',                       SPACE_UID,                       str,        False,   path="/space/href",example_value="46dc9cf1-252f-424b-b52d-5cdd9814987f",transform=lambda x, client:API_VERSION+SPACES+"/"+x),
        MetaProp('COMPUTE',                         COMPUTE,                         dict,       False,   example_value={"name":"name1", "nodes": 1},schema={"name(required)":"string","nodes(optional)":"string"})

    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Runtime Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class LibraryMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    FILEPATH = "filepath"
    VERSION = "version"
    PLATFORM = "platform"
    TAGS = "tags"
    SPACE_UID = "space_uid"
    MODEL_DEFINITION = "model_definition"
    CUSTOM = "custom"
    COMMAND = "command"


    _meta_props_definitions = [
        MetaProp('NAME',                NAME,             str,        True,     "my_lib"),
        MetaProp('DESCRIPTION',         DESCRIPTION,      str,        False,    "my lib"),
        MetaProp('PLATFORM',            PLATFORM,         dict,       True,     {"name": "python", "versions": ["3.5"]},schema={"name(required)":"string","version(required)":"version"}),
        MetaProp('VERSION',             VERSION,          str,        True,     "1.0"),
        MetaProp('FILEPATH',            FILEPATH,         str,        True,     "/home/user/my_lib_1_0.zip"),
        MetaProp('TAGS',                TAGS,             dict,       False,    [{u'value': 'dsx-project.<project-guid>',u'description': 'DSX project guid'}],schema=[{u'value(required)': u'string',u'description(optional)': u'string'}]),
        MetaProp('SPACE_UID',           SPACE_UID,        str,        False,    u'3c1ce536-20dc-426e-aac7-7284cf3befc6', path='/space/href', transform=lambda x, client: API_VERSION+SPACES+"/"+x),
        MetaProp('MODEL_DEFINITION',    MODEL_DEFINITION, bool,       False,     False),
        MetaProp('COMMAND',             COMMAND,          str,        False,     u'command'),
        MetaProp('CUSTOM',              CUSTOM,           dict,       False,     {"field1": "value1"})


    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Custom Libraries')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class SpacesMetaNames(MetaNamesBase):
    TAGS = "tags"
    CUSTOM = "custom"
    NAME = "name"
    DESCRIPTION = "description"
    ONLINE_DEPLOYMENTS = "online_deployments"
    SCHEDULES = "schedules"

    _meta_props_definitions = [
        MetaProp('NAME',                        NAME,               str,      True,   "my_space"),
        MetaProp('TAGS',                        TAGS,               list,     False,  [{u'value': 'dsx-project.<project-guid>',u'description': 'DSX project guid'}],schema=[{u'value(required)': u'string',u'description(optional)': u'string'}]),
        MetaProp('CUSTOM',                      CUSTOM,             dict,     False,  '{"field1":"value1"}'),
        MetaProp('DESCRIPTION',                 DESCRIPTION,        str,      False,  "my_description"),
        MetaProp('ONLINE_DEPLOYMENTS',          ONLINE_DEPLOYMENTS, list,     False,  [{}], path="/deployment_descriptor/online_deployments", schema=[{u'name(optional)': u'string',u'description(optional)': u'string', u'guid(optional)': u'string', u'compute(optional)': {u'name(required)': u'string', u'nodes(optional)': u'number'}}]),
        MetaProp('SCHEDULES',                   SCHEDULES,          list,     False,  [{}],path="/deployment_descriptor/schedules", schema=[{u'cron(optional)': u'string', u'assets(optional)': [{u'name(optional)': u'string', u'description(optional)': u'string', u'guid(optional)': u'string', u'compute(optional)': {u'name(required)': u'string', u'nodes(optional)': u'number'}}]}]),
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Spaces Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class ExportMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    ASSETS = "assets"

    _meta_props_definitions = [
        MetaProp('NAME', NAME, str, True, "my_space"),
        MetaProp('DESCRIPTION', DESCRIPTION, str, False, "my_description"),
        MetaProp('ASSETS', ASSETS, dict, example_value={u'data_assets(optional)': [], u'wml_model': []}, required=True)
    ]
    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Space Exports Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class SpacesPlatformMetaNames(MetaNamesBase):

    NAME = "name"
    DESCRIPTION = "description"
    STORAGE = "storage"
    COMPUTE = "compute"

    _meta_props_definitions = [
        MetaProp('NAME',                        NAME,               str, True,   "my_space"),
        MetaProp('DESCRIPTION',                DESCRIPTION,         str, False,  "my_description"),
        MetaProp('STORAGE', STORAGE, dict, example_value={u'type': 'bmcos_object_storage',
                                                          u'resource_crn': '',
                                                          'delegated(optional)': 'false'}, required=False),
        MetaProp('COMPUTE', COMPUTE, dict, example_value={u'name': 'name',
                                                          u'crn': 'crn of the instance'}, required=False)

    ]
    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Platform Spaces Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class SpacesPlatformMemberMetaNames(MetaNamesBase):
    # IDENTITY = "id"
    # ROLE = "role"
    # IDENTITY_TYPE = "type"
    # STATE = "state"
    MEMBERS = "members"
    MEMBER = "member"
    #
    # _meta_props_definitions = [
    #     MetaProp('IDENTITY',                        IDENTITY,               str, True,   "IBMid-060000123A (service-ID or IAM-userID)"),
    #     MetaProp('ROLE', ROLE, str, True, "Supported values - Viewer/Editor/Admin"),
    #     MetaProp('IDENTITY_USER', IDENTITY_TYPE, str, True, "Supported values - service/user"),
    #     MetaProp('STATE', STATE, str, True, "Supported values - active/pending")
    #
    # ]

    _meta_props_definitions = [
    MetaProp('MEMBERS', MEMBERS, list, False, [{u'id': 'iam-id1', u'role': 'editor', u'type': 'user', u'state': 'active'},
                                               {u'id': 'iam-id2', u'role': 'viewer', u'type': 'user', u'state': 'active'}],
             schema=[{u'id(required)': u'string',
                      u'role(required)': u'string',
                      u'type(required)': u'string',
                      u'state(optional)': u'string'}]),
    MetaProp('MEMBER', MEMBER, dict, False, {u'id': 'iam-id1', u'role': 'editor', u'type': 'user', u'state': 'active'})
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Platform Spaces Member Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class AssetsMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    CONNECTION_ID = "connection_id"
    DATA_CONTENT_NAME = "data_content_name"

    _meta_props_definitions = [
        MetaProp('NAME',                        NAME,               str, True,   "my_data_asset"),
        MetaProp('DATA_CONTENT_NAME', DATA_CONTENT_NAME, str, True, "/test/sample.csv"),
        MetaProp('CONNECTION_ID', CONNECTION_ID, str, False, "39eaa1ee-9aa4-4651-b8fe-95d3ddae"),
        MetaProp('DESCRIPTION',                DESCRIPTION,         str, False,  "my_description")
     ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Data Asset Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


## update this later #Todo
class SwSpecMetaNames(MetaNamesBase):
    TAGS = "tags"
    NAME = "name"
    DESCRIPTION = "description"
    PACKAGE_EXTENSIONS = "package_extensions"
    SOFTWARE_CONFIGURATION = "software_configuration"
    BASE_SOFTWARE_SPECIFICATION = "base_software_specification"

    _meta_props_definitions = [
        MetaProp('NAME',    NAME,   str, True,   "Python 3.6 with pre-installed ML package"),
        MetaProp('DESCRIPTION', DESCRIPTION,    str, False,  "my_description"),
        MetaProp('PACKAGE_EXTENSIONS',  PACKAGE_EXTENSIONS, list, False,  [{"guid":"value"}]),
        MetaProp('SOFTWARE_CONFIGURATION',         SOFTWARE_CONFIGURATION,  dict, False,
                 {"platform": {"name": "python","version": "3.6"}},
                 schema={"platform(required)": "string"}),
        MetaProp('BASE_SOFTWARE_SPECIFICATION',   BASE_SOFTWARE_SPECIFICATION,  dict, True, {"guid": u'BASE_SOFTWARE_SPECIFICATION_ID'} )
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Software Specifications Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class ScriptMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    SOFTWARE_SPEC_UID = "software_spec_uid"

    _meta_props_definitions = [
        MetaProp('NAME',    NAME,   str, True,   "Python script", path='/metadata/name'),
        MetaProp('DESCRIPTION', DESCRIPTION,    str, False,  "my_description", path='/metadata/description'),
        MetaProp('SOFTWARE_SPEC_UID', SOFTWARE_SPEC_UID, str, True, '53628d69-ced9-4f43-a8cd-9954344039a8')
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Script Specifications')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class ShinyMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    SOFTWARE_SPEC_UID = "software_spec_uid"

    _meta_props_definitions = [
        MetaProp('NAME',    NAME,   str, True,   "Shiny App", path='/metadata/name'),
        MetaProp('DESCRIPTION', DESCRIPTION,    str, False,  "my_description", path='/metadata/description'),
        MetaProp('SOFTWARE_SPEC_UID', SOFTWARE_SPEC_UID, str, False, '42c36a39-fcc1-5117-8ff6-1d4523e0d6a6'),
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Shiny Specifications')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class PkgExtnMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    TYPE = "type"

    _meta_props_definitions = [
        MetaProp('NAME',    NAME,   str, True,   "Python 3.6 with pre-installed ML package"),
        MetaProp('DESCRIPTION', DESCRIPTION,    str, False,  "my_description"),
        MetaProp('TYPE',   TYPE,  str, True, "conda_yml/custom_library")
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Package Extensions Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

## update this later #Todo
class HwSpecMetaNames(MetaNamesBase):
    TAGS = "tags"
    NAME = "name"
    DESCRIPTION = "description"
    HARDWARE_CONFIGURATION = "hardware_configuration"
    BASE_SOFTWARE_SPECIFICATION = "base_software_specification"

    _meta_props_definitions = [
        MetaProp('NAME',    NAME,   str, True,   "Python 3.6 with pre-installed ML package"),
        MetaProp('DESCRIPTION', DESCRIPTION,    str, False,  "my_description"),
        MetaProp('HARDWARE_CONFIGURATION',  HARDWARE_CONFIGURATION,  dict, False, {})
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Software Specifications Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class ModelDefinitionMetaNames(MetaNamesBase):
    NAME = "name"
    DESCRIPTION = "description"
    PLATFORM = "platform"
    VERSION = "version"
    SPACE_UID = "space_id"
    COMMAND = "command"
    CUSTOM = "custom"
    _meta_props_definitions = [
        MetaProp('NAME', NAME, str, True, "my_model_definition", path='/metadata/name'),
        MetaProp('DESCRIPTION', DESCRIPTION, str, False, "my model_definition", path='/metadata/description'),
        MetaProp('PLATFORM', PLATFORM, dict, True, {"name": "python", "versions": ["3.5"]},
                 schema={"name(required)": "string", "versions(required)": ["versions"]}),
        MetaProp('VERSION', VERSION, str, True, "1.0"),
        MetaProp('COMMAND', COMMAND, str, False, "python3 convolutional_network.py"),
        MetaProp('CUSTOM', CUSTOM, dict, False, {"field1": "value1"}),
        MetaProp('SPACE_UID', SPACE_UID, str, False, u'3c1ce536-20dc-426e-aac7-7284cf3befc6', path='/space/href',
                 transform=lambda x, client: API_VERSION + SPACES + "/" + x)
    ]
    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Model Definition')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class ConnectionMetaNames(MetaNamesBase):
    DATASOURCE_TYPE = "datasource_type"
    NAME = "name"
    DESCRIPTION = "description"
    PROPERTIES = "properties"
    FLAGS = "flags"

    _meta_props_definitions = [
        MetaProp('NAME',                       NAME,               str, True,   "my_space"),
        MetaProp('DESCRIPTION',                DESCRIPTION,         str, False,  "my_description"),
        MetaProp('DATASOURCE_TYPE', DATASOURCE_TYPE, str, True, "1e3363a5-7ccf-4fff-8022-4850a8024b68"),
        MetaProp('PROPERTIES',                 PROPERTIES, dict, True, example_value={"database": "db_name","host": "host_url","password": "password","username": "user"}),
        MetaProp('FLAGS', FLAGS, list, required=False, example_value="['personal_credentials']")
        ]
    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Spaces Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class DeploymentMetaNames(MetaNamesBase):
    NAME = "name"
    TAGS = "tags"
    DESCRIPTION = "description"
    #ENVIRONMENTS = "environments"
    CUSTOM = "custom"
    AUTO_REDEPLOY = "auto_redeploy"
    COMPUTE = "compute"
    ONLINE = "online"
    BATCH = "batch"
    VIRTUAL = "virtual"
    SPACE_UID = "space"
    HARDWARE_SPEC = "hardware_spec"
    ASSET = "asset"
    R_SHINY = "r_shiny"
    SERVING_NAME = "serving_name"
    HYBRID_PIPELINE_HARDWARE_SPECS = "hybrid_pipeline_hardware_specs"
    OWNER = "owner"

    # VIRTUAL = "virtual"

    _meta_props_definitions = [
        MetaProp('NAME',             NAME,          str,   False, 'my_deployment'),
        MetaProp('TAGS',             TAGS,          list,       False, [{u'value': 'dsx-project.<project-guid>',u'description': 'DSX project guid'}],schema=[{u'value(required)': u'string',u'description(optional)': u'string'}]),
        MetaProp('DESCRIPTION',      DESCRIPTION,   str,   False, 'my_deployment'),
        #MetaProp('ENVIRONMENTS',     ENVIRONMENTS,  list,       False,  ['dev', 'staging']),
        MetaProp('CUSTOM',           CUSTOM,        dict,       False,  {}),
        MetaProp('AUTO_REDEPLOY',    AUTO_REDEPLOY, bool,       False,  False),
        MetaProp('SPACE_UID', SPACE_UID, str, False, u'3c1ce536-20dc-426e-aac7-7284cf3befc6', path='/space/href',transform=lambda x, client: API_VERSION + SPACES + "/" + x),
        MetaProp('COMPUTE', COMPUTE, dict, example_value=None,required=False),
        MetaProp('ONLINE', ONLINE, dict, example_value={}, required=False),
        MetaProp('BATCH', BATCH, dict, example_value={}, required=False),
        MetaProp('VIRTUAL', VIRTUAL, dict, example_value={}, required=False),
        MetaProp('ASSET', ASSET, dict, example_value={}, required=False),
        MetaProp('R_SHINY', R_SHINY, dict, example_value={}, required=False),
        MetaProp('SERVING_NAME', SERVING_NAME, str, False, "deployment", hidden=False, path='/online/parameters/serving_name'),
        MetaProp('HYBRID_PIPELINE_HARDWARE_SPECS', HYBRID_PIPELINE_HARDWARE_SPECS, list, example_value=[{'id': '3342-1ce536-20dc-4444-aac7-7284cf3befc'}], required=False),
        MetaProp('HARDWARE_SPEC',HARDWARE_SPEC, dict, example_value={'id': '3342-1ce536-20dc-4444-aac7-7284cf3befc', 'name': 'S', 'num_nodes': 2}, required=False),
        MetaProp('OWNER', OWNER, str, example_value='<owner_id>', required=False, path='/metadata/owner')
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Deployments Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class DeploymentNewMetaNames(MetaNamesBase):
    NAME = "name"
    TAGS = "tags"
    DESCRIPTION = "description"
    CUSTOM = "custom"
    COMPUTE = "compute"
    ONLINE = "online"
    BATCH = "batch"
    VIRTUAL = "virtual"
    HARDWARE_SPEC = "hardware_spec"
    ASSET = "asset"
    R_SHINY = "r_shiny"
    HYBRID_PIPELINE_HARDWARE_SPECS = "hybrid_pipeline_hardware_specs"
    SERVING_NAME = "serving_name"
    OWNER = "owner"

    # As per https://watson-ml-v4-api.mybluemix.net/wml-restapi-cloud.html#/Deployments/deployments_create
    _meta_props_definitions = [
        MetaProp('TAGS',             TAGS,          list,       False, ['string1', 'string2'], schema=[u'string']),
        MetaProp('NAME',             NAME,          str,   False, 'my_deployment'),
        MetaProp('DESCRIPTION',      DESCRIPTION,   str,   False, 'my_deployment'),
        MetaProp('CUSTOM',           CUSTOM,        dict,       False,  {}),
        MetaProp('ASSET', ASSET, dict, example_value={'id': '4cedab6d-e8e4-4214-b81a-2ddb122db2ab', 'rev': '1'}, required=False),
        MetaProp('HARDWARE_SPEC', HARDWARE_SPEC, dict, example_value={'id': '3342-1ce536-20dc-4444-aac7-7284cf3befc'}, required=False),
        MetaProp('HYBRID_PIPELINE_HARDWARE_SPECS', HYBRID_PIPELINE_HARDWARE_SPECS, list, example_value=[{'node_runtime_id': 'auto_ai.kb', 'hardware_spec': {'id': '3342-1ce536-20dc-4444-aac7-7284cf3befc', 'num_nodes': '2'}}], required=False),
        MetaProp('ONLINE', ONLINE, dict, example_value={}, required=False),
        MetaProp('BATCH', BATCH, dict, example_value={}, required=False),
        MetaProp('R_SHINY', R_SHINY, dict, example_value={"authentication" : "anyone_with_url"}, required=False),
        MetaProp('VIRTUAL', VIRTUAL, dict, example_value={}, required=False),
        MetaProp('SERVING_NAME', SERVING_NAME, str, example_value="deployment", required=False, hidden=True, path='/online/parameters/serving_name'),
        MetaProp('OWNER', OWNER, str, example_value='<owner_id>', required=False, path='/metadata/owner')
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Deployments Specs')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class Migrationv4GACloudMetaNames(MetaNamesBase):
    DESCRIPTION = "description"
    OLD_INSTANCE_ID = "old_instance_id"
    SPACE_ID = "space_id"
    PROJECT_ID = "project_id"
    MODEL_IDS = "model_ids"
    FUNCTION_IDS = "function_ids"
    EXPERIMENT_IDS = "experiment_ids"
    PIPELINE_IDS = "pipeline_ids"
    MAPPING = "mapping"
    SKIP_MIGRATED_ASSETS = "skip_migrated_assets"

    _meta_props_definitions = [
        MetaProp('DESCRIPTION', DESCRIPTION, str, False, "Testing migration", schema=u'string'),
        MetaProp('OLD_INSTANCE_ID', OLD_INSTANCE_ID, str, True, 'df40cf1-252f-424b-b52d-5cdd98143aec', schema=u'string',
                 path="/old_instance/instance_id", transform=lambda x, client: x),
        MetaProp('SPACE_ID', SPACE_ID, str, False, '3fc54cf1-252f-424b-b52d-5cdd9814987f', schema=u'string'),
        MetaProp('PROJECT_ID', PROJECT_ID, str, False, '4fc54cf1-252f-424b-b52d-5cdd9814987f', schema=u'string'),
        MetaProp('MODEL_IDS', MODEL_IDS, list, False, example_value=['afaecb4-254f-689f-4548-9b4298243291'], schema=[u'string']),
        MetaProp('FUNCTION_IDS', FUNCTION_IDS, list, False, example_value=['all'], schema=[u'string']),
        MetaProp('EXPERIMENT_IDS', EXPERIMENT_IDS, list, False, example_value=['ba2ecb4-4542-689a-2548-ab4232b43291'], schema=[u'string']),
        MetaProp('PIPELINE_IDS', PIPELINE_IDS, list, False, example_value=['4fabcb4-654f-489b-9548-9b4298243292'], schema=[u'string']),
        MetaProp('SKIP_MIGRATED_ASSETS', SKIP_MIGRATED_ASSETS, bool, False, example_value=False),
        MetaProp('MAPPING', MAPPING, dict, False, example_value={"dfaecf1-252f-424b-b52d-5cdd98143481": "4fbc211-252f-424b-b52d-5cdd98df310a"})
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('v4ga Cloud migration')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class RemoteTrainingSystemMetaNames(MetaNamesBase):
    TAGS = "tags"
    # SPACE_ID = "space_id"
    # PROJECT_ID = "project_id"
    NAME = "name"
    DESCRIPTION = "description"
    CUSTOM = "custom"
    ORGANIZATION = "organization"
    ALLOWED_IDENTITIES = "allowed_identities"
    REMOTE_ADMIN = "remote_admin"
    DATA_HANDLER = "data_handler"
    LOCAL_TRAINING = "local_training"
    HYPERPARAMS = "hyperparams"
    MODEL = "model"

    _meta_props_definitions = [
        MetaProp('TAGS',             TAGS,          list,       False, ['string1', 'string2'], schema=[u'string']),
        # MetaProp('SPACE_ID', SPACE_ID, str, False, '3fc54cf1-252f-424b-b52d-5cdd9814987f', schema=u'string'),
        # MetaProp('PROJECT_ID', PROJECT_ID, str, False, '4fc54cf1-252f-424b-b52d-5cdd9814987f', schema=u'string'),
        MetaProp('NAME',             NAME,          str,   False, 'my-resource'),
        MetaProp('DESCRIPTION', DESCRIPTION, str, False, "my-resource", schema=u'string'),
        MetaProp('CUSTOM', CUSTOM, dict, False, example_value={"custom_data": "custome_data"}),
        MetaProp('ORGANIZATION', ORGANIZATION, dict, False, example_value={"name": "name", "region": "EU"}),
        MetaProp('ALLOWED_IDENTITIES', ALLOWED_IDENTITIES, list, False, example_value=[{"id": "43689024", "type": "user"}]),
        MetaProp('REMOTE_ADMIN', REMOTE_ADMIN, dict, False, example_value={"id": "id", "type": "user"}),
        MetaProp('DATA_HANDLER', DATA_HANDLER, dict, False, example_value={"info": {"npz_file": "./data_party0.npz"}, "name": "MnistTFDataHandler", "path": "mnist_keras_data_handler"}),
        MetaProp('LOCAL_TRAINING', LOCAL_TRAINING, dict, False, example_value={"name": "LocalTrainingHandler", "path": "ibmfl.party.training.local_training_handler"}),
        MetaProp('HYPERPARAMS', HYPERPARAMS, dict, False, example_value={"epochs": 3, "batch_size": 128}),
        MetaProp('MODEL', MODEL, dict, False, example_value={"info": {"gpu": {"selection": "auto"}}})
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Remote Training System')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

class ExportMetaNames(MetaNamesBase):

    NAME = "name"
    DESCRIPTION = "description"
    ALL_ASSETS = "all_assets"
    ASSET_TYPES = "asset_types"
    ASSET_IDS = "asset_ids"

    _meta_props_definitions = [
        MetaProp('NAME',             NAME,          str,   True, 'my-resource'),
        MetaProp('DESCRIPTION', DESCRIPTION, str, False, "my-resource", schema=u'string'),
        MetaProp('ALL_ASSETS',    ALL_ASSETS, bool, False,  False),
        MetaProp('ASSET_TYPES', ASSET_TYPES, list, False, example_value=["wml_model"]),
        MetaProp('ASSET_IDS', ASSET_IDS, list, False, example_value=["13a53931-a8c0-4c2f-8319-c793155e7517", "13a53931-a8c0-4c2f-8319-c793155e7518"]),
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Export Import metanames')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)

# class EnvironmentMetaNames(MetaNamesBase):
#     NAME = "name"
#     DESCRIPTION = "description"
#     BASE_URL = "base_url"
#     # VIRTUAL = "virtual"
#
#     _meta_props_definitions = [
#         MetaProp('NAME',             NAME,          str,   False, 'my_deployment'),
#         MetaProp('DESCRIPTION',      DESCRIPTION,   str,   False, 'my_deployment'),
#         MetaProp('BASE_URL', BASE_URL, str, False)
#     ]
#
#     __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Environment Specs')
#
#     def __init__(self):
#         MetaNamesBase.__init__(self, self._meta_props_definitions)


class VolumeMetaNames(MetaNamesBase):

    NAME = "name"
    NAMESPACE = "namespace"
    STORAGE_CLASS = "storageClass"
    STORAGE_SIZE = "storageSize"
    EXISTING_PVC_NAME = "existing_pvc_name"
   # MOUNT_PATH = "Mountpath"

    _meta_props_definitions = [
        MetaProp('NAME', NAME, str, True, 'my-volume'),
        MetaProp('NAMESPACE', NAMESPACE, str, True, 'my-volume', schema=u'string'),
        MetaProp('STORAGE_CLASS', STORAGE_CLASS,   str, False,  example_value="nfs-client", schema=u'string'),
        MetaProp('STORAGE_SIZE', STORAGE_SIZE,str, False, example_value=u'2G'),
        #MetaProp('MOUNT_PATH', MOUNT_PATH,str, False, schema=u'string',example_value=""),
        MetaProp('EXISTING_PVC_NAME',EXISTING_PVC_NAME, str, False, example_value="volumes-wml-test-input-2-pvc", schema=u'string')

    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Volume metanames')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)


class FactsheetsMetaNames(MetaNamesBase):

    ASSET_ID = "model_entry_asset_id"
    NAME = "model_entry_name"
    DESCRIPTION = "model_entry_description"
    MODEL_ENTRY_CATALOG_ID = "model_entry_catalog_id"

    _meta_props_definitions = [
        MetaProp('ASSET_ID', ASSET_ID, str, False, '13a53931-a8c0-4c2f-8319-c793155e7517'),
        MetaProp('NAME', NAME, str, False,  example_value="New model entry"),
        MetaProp('DESCRIPTION', DESCRIPTION, str, False, example_value='New model entry'),
        MetaProp('MODEL_ENTRY_CATALOG_ID', MODEL_ENTRY_CATALOG_ID, str, True, example_value='13a53931-a8c0-4c2f-8319-c793155e7517')
    ]

    __doc__ = MetaNamesBase(_meta_props_definitions)._generate_doc('Factsheets metanames')

    def __init__(self):
        MetaNamesBase.__init__(self, self._meta_props_definitions)
