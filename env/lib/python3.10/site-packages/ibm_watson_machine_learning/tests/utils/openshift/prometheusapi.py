#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import requests
from utils.openshift.cli import get_openshift_token
import urllib.parse


class PrometheusAPI:
    def __init__(self, cluster_username, cluster_password, cluster_server_url):
        # self.username = cluster_username
        # self.password = cluster_password
        self.server_url = cluster_server_url

        token = get_openshift_token(
            username=cluster_username,
            password=cluster_password,
            server=cluster_server_url
        )

        self.session = requests.Session()
        self.session.headers.update({"Authorization": "Bearer {}".format(token)})
        self.session.verify = False

        self.prometheus_url = self.__get_prometheus_url()

    def __get_prometheus_url(self):
        response = self.session.get(
            url="{}/apis/monitoring.coreos.com/v1/prometheuses".format(self.server_url)
        )
        if response.status_code != 200:
            print("Unable to get prometheus url!")
            return None

        prometeus_url = response.json()['items'][0]['spec']['externalUrl']

        print("Prometheus url: {}".format(prometeus_url))
        return prometeus_url

    def get_user_details(self):
        response = self.session.get(
            url="{}/apis/user.openshift.io/v1/users/~".format(self.server_url)
        )
        print(response.status_code)
        print(response.text)

    def execute_query(self, query):
        x_url = self.prometheus_url + "api/v1/query?query=" + query
        response = self.session.get(
            url=x_url
        )
        print(response.text)
        print(response.status_code)

        return response.json()

    def get_cpu_sum_auto_ai(self, auto_ai_experiment_id):
        query_str = "pod:container_cpu_usage:sum{pod=~\"train-wml-autoai-" + str(auto_ai_experiment_id) + ".+\"}[2d]"
        query = urllib.parse.quote(query_str)
        return self.execute_query(query)

    def get_mem_sum_auto_ai(self, auto_ai_experiment_id):
        query_str = "pod:container_memory_usage_bytes:sum{pod=~\"train-wml-autoai-" + str(auto_ai_experiment_id) + ".+\"}[2d]"
        query = urllib.parse.quote(query_str)
        return self.execute_query(query)