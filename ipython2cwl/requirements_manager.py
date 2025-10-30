from typing import List
import subprocess
import sys


class RequirementsManager:
    """
    That class is responsible for generating the requirements.txt file of the activated python environment.
    """

    @classmethod
    def get_all(cls) -> List[str]:
        try:
            # Use pip freeze command as the modern approach
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
            )
            lines = result.stdout.strip().split("\n")
            return [
                line for line in lines if line and not line.startswith("ipython2cwl")
            ]
        except Exception:
            # Fallback to pkg_resources if pip freeze fails
            try:
                import pkg_resources

                return [
                    str(package.as_requirement())
                    for package in pkg_resources.working_set
                    if package.project_name != "ipython2cwl"
                ]
            except Exception:
                return []
