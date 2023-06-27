#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import sys
import argparse
import re
import pip

def get_required_packages(filename):

    def parse_package_name(package_name):
        package_name_split = re.split('>=|<=', package_name)
        package_name_clean = package_name_split[0]
        if len(package_name_split) > 1:
            print('Version restrictions "{}" found in {} are not supported'.format(', '.join(package_name_split[1:]), package_name_clean))
        if '==' in package_name_clean:
            pkg = dict(zip(['name', 'version'], package_name_clean.split('==')))
        else:
            pkg = {'name': package_name_clean, 'version': None}
        return pkg

    try:
        with open(filename) as f:
            packages_from_requirements = f.read().splitlines()
    except IOError:
        print('Could not read file: {}'.format(filename))

    return [parse_package_name(pkg) for pkg in packages_from_requirements]

def get_installed_packages():
    return [{'name': pkg.project_name, 'version': pkg.version} for pkg in pip.get_installed_distributions()]

def get_missing_packages(installed_pkgs, required_pkgs):

    def infer_action(pkg):
        installed_pkg = None
        for p in installed_pkgs:
            if p['name'] == pkg['name']:
                installed_pkg = p

        if installed_pkg is None:
            return {'action': 'install'}
        else:
            if (pkg['version'] is not None) and (pkg['version'] != installed_pkg['version']):
                    return {'action': 'reinstall'}
        return {'action': None}

    missing_pkgs = []
    for pkg in required_pkgs:
        pkg_action = infer_action(pkg)
        if pkg_action['action'] is not None:
            missing_pkgs.append(dict(pkg, **pkg_action))

    return missing_pkgs

def run(filename=None):

    required_packages = get_required_packages(filename)
    installed_packages = get_installed_packages()
    missing_packages = get_missing_packages(installed_packages, required_packages)

    if len(missing_packages) > 0:
        if 'win' in sys.platform:
            print('Automatic installation is not supported on Windows')
            print('Check if following packages are installed: \n{}'.format('\n'.join(packages_from_requirements)))
            quit()

        for pkg in missing_packages:
            if pkg['action'] == 'reinstall':
                pip_uninstall_arg = ['uninstall']
                pip_uninstall_arg.append('{}'.format(pkg['name']))
                pip_uninstall_arg.append('-y')
                pip.main(pip_uninstall_arg)
            pip_install_arg = ['install']
            if pkg['version'] is None:
                pip_install_arg.append('{}'.format(pkg['name']))
            else:
                pip_install_arg.append('{}=={}'.format(pkg['name'], pkg['version']))
            pip.main(pip_install_arg)
    else:
        print("All required packages are installed.")

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-r', type=str, default='requirements.txt', help='Requirements file name')
    FLAGS, unparsed = arg_parser.parse_known_args()
    run(filename=FLAGS.r)
