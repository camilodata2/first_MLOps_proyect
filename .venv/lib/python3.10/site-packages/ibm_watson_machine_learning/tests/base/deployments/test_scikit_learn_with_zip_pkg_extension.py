#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import filecmp
import os
import unittest

from sklearn import datasets

from ibm_watson_machine_learning.tests.base.abstract.abstract_online_deployment_test import AbstractOnlineDeploymentTest


class TestScikitLearnDeployment(AbstractOnlineDeploymentTest, unittest.TestCase):
    """
    Test case checking the scenario of storing & deploying scikit-learn
    using compressed file.
    """
    deployment_type = "scikit-learn_1.1"
    software_specification_name = "runtime-22.2-py3.10"
    model_name = deployment_name = "sklearn_model_from_gz_with_pkg_extension"
    file_name = 'scikit_model.tar.gz'
    IS_MODEL = True

    def get_model(self):
        model_data = datasets.load_digits()
        TestScikitLearnDeployment.training_data = model_data.data
        TestScikitLearnDeployment.training_target = model_data.target

        return os.path.join(os.getcwd(), 'base', 'artifacts', 'scikit-learn', self.file_name)

    def get_package_extension_file(self):
        return os.path.join(os.getcwd(), 'base', 'artifacts', 'other', 'pkg_extension.zip')

    def create_package_extension(self):
        meta_prop_pkg_extn = {
            self.wml_client.package_extensions.ConfigurationMetaNames.NAME: "test pkg_extension env",
            self.wml_client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "Environment with test pkg_extension",
            self.wml_client.package_extensions.ConfigurationMetaNames.TYPE: "pip_zip"
        }

        pkg_extn_details = self.wml_client.package_extensions.store(
            meta_props=meta_prop_pkg_extn,
            file_path=self.get_package_extension_file()
        )
        pkg_extn_uid = self.wml_client.package_extensions.get_uid(pkg_extn_details)

        self.wml_client.package_extensions.download(pkg_extn_uid, "./test_pkg_extension.zip")

        self.assertTrue(filecmp.cmp('./test_pkg_extension.zip', self.get_package_extension_file()))

        return pkg_extn_uid

    def create_software_spec_with_pkg_extension(self, pkg_extn_id):
        sw_spec_meta_props = {
            self.wml_client.software_specifications.ConfigurationMetaNames.NAME: "runtime-22.2-py3.10 with test pkg_extension",
            self.wml_client.software_specifications.ConfigurationMetaNames.PACKAGE_EXTENSIONS: [{'guid': pkg_extn_id}],
            self.wml_client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {
                'guid': self.wml_client.software_specifications.get_id_by_name(self.software_specification_name),
            }
        }
        
        # Note: Delete old software spec if the software spec with the same name already exists:
        old_sw_spec_id = self.wml_client.software_specifications.get_id_by_name(sw_spec_meta_props['name'])
        if old_sw_spec_id and old_sw_spec_id.lower() != 'not found':
            self.wml_client.software_specifications.delete(old_sw_spec_id)
        # end note

        sw_spec_details = self.wml_client.software_specifications.store(sw_spec_meta_props)
        sw_spec_uid = self.wml_client.software_specifications.get_uid(sw_spec_details)

        return sw_spec_uid

    def create_model_props(self):
        pkg_extn_uid = self.create_package_extension()
        sw_spec_uid = self.create_software_spec_with_pkg_extension(pkg_extn_uid)

        return {
            self.wml_client.repository.ModelMetaNames.NAME: self.model_name,
            self.wml_client.repository.ModelMetaNames.TYPE: self.deployment_type,
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid,
        }

    def create_deployment_props(self):

        deployment_props = {
            self.wml_client.deployments.ConfigurationMetaNames.NAME: self.deployment_name,
            self.wml_client.deployments.ConfigurationMetaNames.ONLINE: {}
        }
        if self.wml_client.ICP:
            deployment_props.update(
                {
                    self.wml_client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {
                        "name": "M",
                        "num_nodes": 1
                    }
                }
            )

        return deployment_props

    def create_scoring_payload(self):
        return {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                'values': [
                    [0.0, 0.0, 5.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     12.0, 15.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 15.0, 16.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 13.0,
                     16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 16.0, 8.0,
                     0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 15.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 6.0, 16.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 13.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 5.0, 5.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     13.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 16.0, 9.0, 4.0, 1.0, 0.0, 0.0, 3.0, 16.0, 16.0, 16.0,
                     16.0, 10.0, 0.0, 0.0, 5.0, 16.0, 11.0, 9.0, 6.0, 2.0]]
            }]
        }

    def test_01_store_model(self):
        model_props = self.create_model_props()
        TestScikitLearnDeployment.model = self.get_model()

        model_details = self.wml_client.repository.store_model(
            meta_props=model_props,
            model=self.model,
            training_data=self.training_data,
            training_target=self.training_target
        )
        TestScikitLearnDeployment.model_id = self.wml_client.repository.get_model_id(model_details)
        self.assertIsNotNone(self.model_id)


if __name__ == "__main__":
    unittest.main()
