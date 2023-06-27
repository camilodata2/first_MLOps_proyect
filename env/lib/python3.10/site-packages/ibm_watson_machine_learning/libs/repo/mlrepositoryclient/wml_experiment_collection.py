#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging, re

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.experiment_artifact import ExperimentArtifact
from ibm_watson_machine_learning.libs.repo.swagger_client.api_client import ApiException
import json
from .wml_experiment_adapter import WmlExperimentCollectionAdapter
from ibm_watson_machine_learning.libs.repo.swagger_client.models import ExperimentInput,TagRepository,\
    ExperimentInputSettings,MetricObjectExperiments,HyperParametersOptimizationExperiments,EvaluationDefinitionExperiments
from ibm_watson_machine_learning.libs.repo.swagger_client.models import TrainingReferenceExperiments,ComputeConfigurationExperiments
from ibm_watson_machine_learning.libs.repo.swagger_client.models import  ConnectionObjectTargetExperiments,ConnectionObjectSourceExperiments,AuthorExperiments, PatchOperationExperiments, HyperParametersExperiments
from ibm_watson_machine_learning.libs.repo.swagger_client.models import HyperParametersOptimizationExperimentsMethodParameters, HyperParametersOptimizationExperimentsMethod, \
    HyperParametersExperimentsDoubleRange, HyperParametersExperimentsIntRange

logger = logging.getLogger('WmlExperimentCollection')


class WmlExperimentCollection:
    """
    Client operating on experiments in repository service.

    :param str base_path: base url to Watson Machine Learning instance
    :param MLRepositoryApi repository_api: client connecting to repository rest api
    :param MLRepositoryClient client: high level client used for simplification and argument for constructors
    """
    def __init__(self, base_path, repository_api, client):

        from ibm_watson_machine_learning.libs.repo.mlrepositoryclient import MLRepositoryClient
        from ibm_watson_machine_learning.libs.repo.mlrepositoryclient import MLRepositoryApi
        if not isinstance(base_path, str) and not isinstance(base_path, unicode):
            raise ValueError('Invalid type for base_path: {}'.format(base_path.__class__.__name__))

        if not isinstance(repository_api, MLRepositoryApi):
            raise ValueError('Invalid type for repository_api: {}'.format(repository_api.__class__.__name__))

        if not isinstance(client, MLRepositoryClient):
            raise ValueError('Invalid type for client: {}'.format(client.__class__.__name__))

        self.base_path = base_path
        self.repository_api = repository_api
        self.client = client


    def all(self, queryMap=None):
        """
        Gets info about all experiments which belong to this user.

        Not complete information is provided by all(). To get detailed information about experiment use get().

        :return: info about experiments
        :rtype: list[ExperimentsArtifact]
        """
        all_experiments = self.repository_api.repository_listexperiments(queryMap)
        list_experiment_artifact = []
        if all_experiments is not None:
            resr = all_experiments.resources

            for iter1 in resr:
                list_experiment_artifact.append(WmlExperimentCollectionAdapter(iter1, self.client).artifact())
            return list_experiment_artifact
        else:
            return []


    def get(self, experiment):

        """
        Gets detailed information about experiment.

        :param str experiment_id: uid used to identify experiment
        :return: returned object has all attributes of SparkPipelineArtifact but its class name is PipelineArtifact
        :rtype: PipelineArtifact(SparkPipelineLoader)
        """

        if not isinstance(experiment, str) and  not isinstance(experiment, unicode):
            raise ValueError('Invalid type for experiment_id: {}'.format(experiment.__class__.__name__))
        if(experiment.__contains__("/v3/experiments")):
            matched = re.search('.*/v3/experiments/([A-Za-z0-9\-]+)', experiment)
            if matched is not None:
                experiment_id = matched.group(1)
                return self.get(experiment_id)
            else:
                raise ValueError('Unexpected artifact href: {} format'.format(experiment))
        else:
            experiment_output = self.repository_api.v3_experiments_id_get(experiment)
            if experiment_output is not None:
                return WmlExperimentCollectionAdapter(experiment_output,self.client).artifact()
            else:
                raise Exception('Experiment not found'.format(experiment))


    def remove(self, experiment):
        """
        Removes experiment with given experiment_id.

        :param str experiment_id: uid used to identify experiment
        """

        if not isinstance(experiment, str) and not isinstance(experiment, unicode):
            raise ValueError('Invalid type for experiment_id: {}'.format(experiment.__class__.__name__))
        if(experiment.__contains__("/v3/experiments")):
            matched = re.search('.*/v3/experiments/([A-Za-z0-9\-]+)', experiment)
            if matched is not None:
                experiment_id = matched.group(1)
                self.remove(experiment_id)
            else:
                raise ValueError('Unexpected experiment artifact href: {} format'.format(experiment))
        else:
            return self.repository_api.v3_experiments_id_delete(experiment)

    def patch(self, experiment_id, artifact):
        experiment_patch_input = self.prepare_experiment_patch_input(artifact)
        experiment_patch_output = self.repository_api.v3_experiments_id_patch_with_http_info(experiment_id, experiment_patch_input)
        statuscode = experiment_patch_output[1]

        if statuscode != 200:
            logger.info('Error while patching experiment: no location header')
            raise ApiException(statuscode,"Error while patching experiment")

        if experiment_patch_output is not None:
            new_artifact =  WmlExperimentCollectionAdapter(experiment_patch_output[0],self.client).artifact()
        return new_artifact

    def save(self, artifact):
        """
        Saves experiment in repository service.

        :param SparkPipelineArtifact artifact: artifact to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: SparkPipelineArtifact
        """
        logger.debug('Creating a new WML experiment: {}'.format(artifact.name))

        if not issubclass(type(artifact), ExperimentArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        experiment_input = self._prepare_wml_experiment_input(artifact)
        experiment_output = self.repository_api.wml_assets_experiment_creation_with_http_info(experiment_input)


        statuscode = experiment_output[1]
        if statuscode != 201:
            logger.info('Error while creating experiment: no location header')
            raise ApiException(statuscode, 'No artifact location')

        if experiment_output is not None:
            new_artifact =  WmlExperimentCollectionAdapter(experiment_output[0],self.client).artifact()
        return new_artifact



    @staticmethod
    def _prepare_wml_experiment_input(artifact):

        tags_data_list = None
        settings_data = None
        training_reference_list=None
        training_data_ref = None
        training_results_ref=None
        hyper_param_list = None

        #tags
        tags=artifact.meta.prop(MetaNames.EXPERIMENTS.TAGS)
        if isinstance(tags, str):
            tags_list = json.loads(artifact.meta.prop(MetaNames.EXPERIMENTS.TAGS))
            tags_data_list = []
            if isinstance(tags_list, list):
               for iter1 in tags_list:
                 tags_data = TagRepository()
                 for key in iter1:
                    if key == 'value':
                        tags_data.value= iter1['value']
                    if key == 'description':
                        tags_data.description = iter1['description']
                 tags_data_list.append(tags_data)
            else:
                raise ValueError("Invalid tag Input")

        #settings
        if artifact.meta.prop(MetaNames.EXPERIMENTS.SETTINGS) is  None:
           raise ValueError("MetaNames.EXPERIMENTS.SETTINGS not defined")

        settings=artifact.meta.prop(MetaNames.EXPERIMENTS.SETTINGS)
        if isinstance(settings, str):
            settings = json.loads(artifact.meta.prop(MetaNames.EXPERIMENTS.SETTINGS))

            if isinstance(settings, dict):
                author=settings.get('author', None)
                author_experiment = None
                if author is not None:
                    author_name = author.get('name',None)
                    author_experiment = AuthorExperiments(author_name)


                evaluation_definition = settings.get('evaluation_definition',None)
                evaluation_definition_exp = None
                if evaluation_definition is not None:
                    evaluation_definition_method = evaluation_definition.get('method',None)
                    evaluation_definition_metrics = evaluation_definition.get('metrics')
                    metrics_experiments = []
                    if isinstance(evaluation_definition_metrics, list):
                      for iter1 in evaluation_definition_metrics:
                        metrics_data = MetricObjectExperiments()
                        for key in iter1:
                            if key == 'name':
                                metrics_data.name= iter1['name']
                            if key == 'maximize':
                                metrics_data.maximize= iter1['maximize']
                            metrics_experiments.append(metrics_data)
                    else:
                        raise ValueError("Invalid Input: Metrics list Expected")
                    evaluation_definition_exp = EvaluationDefinitionExperiments(evaluation_definition_method,metrics_experiments)


                settings_data = ExperimentInputSettings(
                    name = settings.get('name'),
                    description = settings.get('description', None),
                    author=author_experiment,
                    label_column= settings.get('label_column', None),
                    evaluation_definition= evaluation_definition_exp
                    )

        #training_refrences
        training_ref=artifact.meta.prop(MetaNames.EXPERIMENTS.TRAINING_REFERENCES)
        if isinstance(training_ref, str):
            training_ref_list = json.loads(artifact.meta.prop(MetaNames.EXPERIMENTS.TRAINING_REFERENCES))
            training_reference_list = []
            if isinstance(training_ref_list, list):
                for iter1 in training_ref_list:
                    compute = iter1.get('compute_configuration', None)
                    compute_config_object = None
                    if compute is not None:
                        compute_name = compute.get('name')
                        compute_nodes = compute.get('nodes',None)
                        compute_config_object = ComputeConfigurationExperiments(compute_name, compute_nodes)

                    hyper_params_optimization = iter1.get('hyper_parameters_optimization', None)
                    hyper_parameters_optimization_object = None

                    if hyper_params_optimization is not None:

                        hyper_parameters_optimization_method = hyper_params_optimization.get('method')
                        method_name = hyper_parameters_optimization_method.get('name', None)

                        parameters = hyper_parameters_optimization_method.get('parameters', None)
                        params_list = []
                        if parameters is not None:

                            if isinstance(parameters, list):

                                for iteration in parameters:
                                    param_name = None
                                    param_string_value = None
                                    param_double_value = None
                                    param_int_value = None
                                    for key in iteration:

                                        if(key == 'name'):
                                            param_name = iteration['name']
                                        if(key == 'string_value'):
                                            param_string_value = iteration['string_value']
                                        if(key == 'double_value'):
                                            param_double_value = iteration['double_value']
                                        if(key == 'int_value'):
                                           param_int_value = iteration['int_value']
                                    parameters_object = HyperParametersOptimizationExperimentsMethodParameters(param_name, param_string_value, param_double_value, param_int_value)
                                    params_list.append(parameters_object)
                        hyper_parameters_optimization_method_object = HyperParametersOptimizationExperimentsMethod(method_name, params_list)

                        hyper_params = hyper_params_optimization.get('hyper_parameters', None)
                        if hyper_params is not None:
                            if isinstance(hyper_params, list):
                                hyper_param_list = []
                                double_value_list = []
                                int_value_list = []
                                string_value_list = []
                                double_range_object = None
                                int_range_object = None
                                for iter2 in hyper_params:
                                    for key in iter2:

                                        if key == 'name':

                                            hyper_params_name= iter2['name']
                                        if key == 'double_values':
                                            double_values = iter2['double_values']
                                            if double_values is not None:
                                                if isinstance(double_values, list):

                                                    for double_value in double_values:

                                                        double_value_list.append(double_value)

                                        if key == 'int_values':
                                            int_values = iter2['int_values']
                                            if int_values is not None:
                                                if isinstance(int_values, list):

                                                    for int_value in int_values:

                                                        int_value_list.append(int_value)

                                        if key == 'string_values':

                                            string_values = iter2['string_values']
                                            if string_values is not None:
                                                if isinstance(string_values, list):

                                                    for value in string_values:

                                                        string_value_list.append(value)

                                        if key == 'double_range':
                                            double_values_range = iter2['double_range']
                                            if double_values_range is not None:
                                                double_min_value = double_values_range.get('min_value')
                                                double_max_value = double_values_range.get('max_value')
                                                double_step = double_values_range.get('step', None)
                                                double_power = double_values_range.get('power', None)
                                                double_range_object = HyperParametersExperimentsDoubleRange(
                                                    double_min_value, double_max_value, double_step, double_power)

                                        if key == 'int_range':


                                            int_values_range = iter2['int_range']
                                            if int_values_range is not None:
                                                int_min_value = int_values_range.get('min_value')
                                                int_max_value = int_values_range.get('max_value')
                                                int_step = int_values_range.get('step', None)
                                                int_power = int_values_range.get('power', None)
                                                int_range_object = HyperParametersExperimentsIntRange(
                                                    int_min_value, int_max_value, int_step, int_power)

                                    if not double_value_list:
                                        double_value_list=None
                                    if not int_value_list:
                                        int_value_list=None

                                    hyper_params_object = HyperParametersExperiments(hyper_params_name,
                                                                                     double_value_list,
                                                                                     int_value_list,
                                                                                     string_value_list,
                                                                                     double_range_object,
                                                                                     int_range_object)
                                    hyper_param_list.append(hyper_params_object)
                        hyper_parameters_optimization_object = HyperParametersOptimizationExperiments(hyper_parameters_optimization_method_object, hyper_param_list)

                    training_reference = TrainingReferenceExperiments(
                            name = iter1.get('name'),
                            training_definition_url= iter1.get('training_definition_url'),
                            command = iter1.get('command', None),
                            hyper_parameters_optimization = hyper_parameters_optimization_object,
                            compute_configuration = compute_config_object,
                            pretrained_model_url = iter1.get('pretrained_model_url', None)
                        )
                    training_reference_list.append(training_reference)

            else:
                raise ApiException(404, 'Invalid Input')

        #Training_data_reference
        if artifact.meta.prop(MetaNames.EXPERIMENTS.TRAINING_DATA_REFERENCE) is  None:
            raise ValueError("MetaNames.EXPERIMENTS.TRAINING_DATA_REFERENCE not defined")

        training_data_ref = artifact.meta.prop(MetaNames.EXPERIMENTS.TRAINING_DATA_REFERENCE)
        if isinstance(training_data_ref, str):
            dataref = json.loads(artifact.meta.prop(MetaNames.EXPERIMENTS.TRAINING_DATA_REFERENCE))
            if isinstance(dataref, dict):
                training_data_ref = ConnectionObjectSourceExperiments(
                        dataref.get('type'),
                        dataref.get('connection'),
                        dataref.get('source')
                    )
            else:
                raise ApiException(404, 'Invalid  TRAINING_DATA_REFERENCE Input')

        #Training_result_refernce
        if artifact.meta.prop(MetaNames.EXPERIMENTS.TRAINING_RESULTS_REFERENCE) is not None:
            training_results_ref = artifact.meta.prop(MetaNames.EXPERIMENTS.TRAINING_RESULTS_REFERENCE)
            if isinstance(training_results_ref, str):
                resultref_dict = json.loads(artifact.meta.prop(MetaNames.EXPERIMENTS.TRAINING_RESULTS_REFERENCE))
                if isinstance(resultref_dict, dict):
                    training_results_ref = ConnectionObjectTargetExperiments(
                        resultref_dict.get('type', None),
                        resultref_dict.get('connection', None),
                        resultref_dict.get('target', None)
                    )
                else:
                    raise ApiException(404, 'Invalid TRAINING_RESULTS_REFERENCE Input')


        experiments_input = ExperimentInput(
            tags=tags_data_list,
            settings=settings_data,
            training_references=training_reference_list,
            training_data_reference=training_data_ref,
            training_results_reference=training_results_ref

        )

        return experiments_input


    @staticmethod
    def prepare_experiment_patch_input(artifact):
        patch_list =[]
        patch_input = artifact.meta.prop(MetaNames.EXPERIMENTS.PATCH_INPUT)
        if isinstance(patch_input, str):
            patch_input_list = json.loads(artifact.meta.prop(MetaNames.EXPERIMENTS.PATCH_INPUT))
            if isinstance(patch_input_list, list):
                for iter1 in patch_input_list:
                    experiment_patch = PatchOperationExperiments(
                        op = iter1.get('op'),
                        path= iter1.get('path'),
                        value = iter1.get('value', None),
                        _from =iter1.get('from', None),
                    )
                    patch_list.append(experiment_patch)

                return patch_list
            else:
                raise ApiException(404, 'Invalid Patch Input')
        else:
            raise ApiException(404, 'Invalid Patch Input')
