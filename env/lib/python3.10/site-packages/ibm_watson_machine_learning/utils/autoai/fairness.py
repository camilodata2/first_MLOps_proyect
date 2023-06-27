#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _get_fairness_details(run_details, pipeline_name):
    """Return fairness details based on experiment run details.

    :param run_details: details of run
    :type run_details: dict
    :param pipeline_name: name of the pipeline
    :type pipeline_name: str

    :return: extracted fairness details
    :rtype: dict
    """
    for pipeline in run_details['entity']['status'].get('metrics', []):
        if pipeline['context']['intermediate_model']['name'].split('P')[-1] == pipeline_name.split('_')[-1]:
            return pipeline['context']['fairness']


def get_pipeline_fairness_score(fairness_details):
    return fairness_details['visualization_data']['holdout']['monitored_group']['fairness']['value']


def get_favorable_outcomes(fairness_details):
    return fairness_details['visualization_data']['holdout']['monitored_group']['favorable_outcome']['values']


def _get_pipeline_percentage_of(fairness_details, group, outcome):
    return fairness_details['visualization_data']['holdout'][group][outcome]['percentage']


def _get_pipeline_perfect_equality(fairness_details):
    return fairness_details['visualization_data']['holdout']['monitored_group']['favorable_outcome']['perfect_equality']['percentage']


def get_attributes_scores(fairness_details):
    return {attr['feature']: attr['fairness']['percentage'] for attr in fairness_details['visualization_data']['holdout']['monitored_group']['protected_attributes']}


def _get_attributes_perfect_equality(fairness_details):
    return {attr['feature']: attr['favorable_outcome']['perfect_equality']['percentage'] for attr in fairness_details['visualization_data']['holdout']['monitored_group']['protected_attributes']}


def _get_pipeline_fairness(fairness_details):
    """Return pandas dataframe with fairness descriptive statics for pipeline model.

    :param fairness_details: details of fairness
    :type fairness_details: dict

    :return: dataframe containing fairness outcome
    :rtype: pandas df
    """

    favorable_outcomes_percentage_monitored = _get_pipeline_percentage_of(fairness_details, 'monitored_group', 'favorable_outcome')
    favorable_outcomes_percentage_reference = _get_pipeline_percentage_of(fairness_details, 'reference_group', 'favorable_outcome')

    data_model = [['favorable', 'monitored', favorable_outcomes_percentage_monitored],
                  ['favorable', 'reference', favorable_outcomes_percentage_reference],
                  ['unfavorable', 'monitored', 100 - favorable_outcomes_percentage_monitored],
                  ['unfavorable', 'reference', 100 - favorable_outcomes_percentage_reference]]

    return pd.DataFrame(data_model, columns=['Outcome', 'Group', '%'])


def get_protected_attributes(fairness_details):
    return [attr['feature'] for attr in fairness_details['visualization_data']['holdout']['monitored_group']['protected_attributes']]


def _get_attributes_fairness(fairness_details):
    attr_data = []

    for group in ['monitored_group', 'reference_group']:
        attributes = fairness_details['visualization_data']['holdout'][group]['protected_attributes']
        for attr in attributes:
            name = attr['feature']

            if len(attributes) > 1:
                fav_percentage = attr['favorable_outcome']['percentage']
            else:
                fav_percentage = _get_pipeline_percentage_of(fairness_details, group, 'favorable_outcome')

            attr_data.append(['unfavorable', group, name, 100 - fav_percentage])
            attr_data.append(['favorable', group, name, fav_percentage])

    return pd.DataFrame(attr_data, columns=['Outcome', 'Group', 'Feature', '%'])


def visualize(run_details: dict, pipeline_name: str):
    """Plot favorable outcome distributions using plotly package.

    :param run_details: details of run
    :type run_details: dict
    :param pipeline_name: name of pipeline
    :type pipeline_name: str
    """
    if not isinstance(run_details, dict):
        raise ValueError('run_details should be dict not {0}'.format(type(run_details)))

    if not isinstance(pipeline_name, str):
        raise ValueError('pipeline_name should be str not {0}'.format(type(pipeline_name)))

    colors = ["#2A66DE", "gray"]
    fairness_details = _get_fairness_details(run_details=run_details, pipeline_name=pipeline_name)
    df = _get_pipeline_fairness(fairness_details)
    pipeline_equality = _get_pipeline_perfect_equality(fairness_details)
    df_attr = _get_attributes_fairness(fairness_details)
    fig1 = px.bar(df, y="Group", x="%", color="Outcome", title="Pipeline", orientation='h', color_discrete_sequence=colors)
    fig1.add_vline(x=pipeline_equality, line_width=1, line_color="black", annotation_text="Perfect equality", annotation_position="top left")
    fig1.show()

    fig2 = go.Figure()
    fig2.update_layout(
        title="Protected attributes",
        template="simple_white",
        xaxis=dict(title_text="Outcome [%]"),
        yaxis=dict(title_text="Protected attribute"),
        barmode="stack"
    )

    for r, c in zip(sorted(df_attr.Outcome.unique()), colors):
        plot_df = df_attr[df_attr.Outcome == r]
        fig2.add_trace(go.Bar(y=[plot_df.Feature, plot_df.Group], x=plot_df['%'], name=r, marker_color=c, orientation='h'))
    fig2.show()
