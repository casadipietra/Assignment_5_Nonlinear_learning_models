# downloadrequirements.py

import subprocess
import sys
import pkg_resources
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def install_if_needed(requirements_file="requirements.txt"):
    with open(requirements_file) as f:
        requirements = f.read().splitlines()

    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    for requirement in requirements:
        pkg_name = requirement.split("==")[0].split(">=")[0].split("<")[0].lower()
        installed_version = installed_packages.get(pkg_name)
        
        try:
            pkg_resources.require(requirement)
            print(f"✅ {requirement} déjà satisfait (version installée : {installed_version})")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            print(f"🔄 Installation ou rétrogradation de {requirement} (version installée : {installed_version})")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])

# Permet l'appel direct
if __name__ == "__main__":
    install_if_needed()
