import sys
import os
from itertools import chain
from pathlib import Path
from pprint import pprint
from shutil import rmtree
from typing import Any, Dict, List, Optional, Union

HERE = Path(__file__).parent.resolve()
TEMPLATE = HERE / "_templates"
TMPDIR = HERE / "_tmp"
WORKFLOWS = TEMPLATE / ".github" / "workflows"

checkers = ["mypy", "flake8", "pydocstyle"]
formatters = ["black", "isort"]

files: Dict[Path, Optional[List[str]]] = {
    TEMPLATE / "docs": ["docs"],
    TEMPLATE / "examples": ["docs"],
    TEMPLATE / "tests": ["testing"],
    TEMPLATE / ".flake8": ["flake8"],
    TEMPLATE / ".pre-commit-config.yaml": ["pre-commit"],
    TEMPLATE / "pyproject.toml": formatters + ["mypy", "pydocstyle", "testing"],
    TEMPLATE / "LICENSE.txt": ["license"],
    TEMPLATE / "CITATION.cff": ["citation"],
    WORKFLOWS / "citation_cff.yml": ["citations"],
    WORKFLOWS / "docs.yml": ["docs"],
    WORKFLOWS / "pre-commit.yml": ["pre-commit"],
    WORKFLOWS / "pytest.yml": ["testing"],
    TEMPLATE / "src" / "py.typed": ["mypy"],
    TEMPLATE / "src" / "__init__.py": None,
    TEMPLATE / "src" / "myfile.py": None,
    TEMPLATE / "setup.py": None,
    TEMPLATE / "Makefile": None,
    TEMPLATE / "MANIFEST.in": None,
    TEMPLATE / "README.md": None,
    TEMPLATE / ".gitignore": None,
}

options: Dict[str, Any] = {
    "features": {
        "testing": {
            "default": True,
            "prompt": "Would you like testing?",
            "help": (
                "Will use `pytest` for testing as well as set up github workflows"
                " that run automatically when pushes and PR are made. Further enables"
                " automatic code coverage report on pull requests as well as when"
                " working locally."
            ),
        },
        "docs": {
            "default": False,
            "prompt": "Would you like `documentation`?",
            "help": (
                "This will install `sphinx` and ready the repo for building"
                " documentation. This includes are own `automl_sphinx_theme`,"
                " examples building and pushing to github pages so others can view"
                " your documentation. It also enables a github workflow to verify that"
                " the docs can be built."
            ),
        },
        "packaging": {
            "default": False,
            "prompt": "Would you like `packaging`?",
            "help": (
                "This will enable the repo to be more ready for packaging to PyPI."
                " If you plan to have this repo as a package that people can install"
                " through `pip install ...` then enable this option."
            ),
        },
        "mypy": {
            "default": False,
            "prompt": "Would you like `mypy`?",
            "help": (
                "Will install `mypy` which is a static type checked for Python. Python"
                " is normally a untyped language like Javascript where as languages like"
                " Java and C++ requires types."
                "\n\nIf you are not used to writing code with types, we recommend to keep"
                " this disabled."
            ),
        },
        "flake8": {
            "default": True,
            "prompt": "Would you like `flake8`?",
            "help": (
                "Will install `flake8` which does some static code checking to find simple"
                " programmer errors such as variables not existing, variables not imported"
                " and syntax errors."
            ),
        },
        "pydocstyle": {
            "default": True,
            "prompt": "Would you like `pydocstyle`?",
            "help": (
                "Will install `pydocstyle` which ensures code is well documented and"
                " correctly formatted. We use the 'numpy' code style but this can be"
                " changed later."
            ),
        },
        "black": {
            "default": True,
            "prompt": "Would you like `black`?",
            "help": (
                "Will install `black` which is a code formatter. It can be run to"
                " automatically format you code to a stricter version of PEP8 style."
                " This helps keep code style consistent and prevents you needing to"
                " worry about formatting, focusing on just writing code."
            ),
        },
        "isort": {
            "default": True,
            "prompt": "Would you like `isort`?",
            "help": (
                "Will install `isort` which formats imports. It can be run to"
                " automatically sort imports and section them nicely. This means import"
                " styles are consistent and again lets you not worry about formatting"
                " imports or their order."
            ),
        },
        "pre-commit": {
            "default": False,
            "prompt": "Would you like `pre-commit`?",
            "help": (
                "Will install `pre-commit` which runs all the previously installed tools"
                " automatically every time a commit is done. This also enables a github"
                " workflows which will do these same checks."
            ),
        },
        "citations": {
            "default": False,
            "prompt": "Would you like a `CITATION.cff`?",
            "help": (
                "This will copy over a `CITATION.cff` file. If you plan to publish your"
                " code for a publication, we highly recommend this. It also enables a"
                " a workflow which will validate that it is correct."
            ),
        },
        "license": {
            "default": False,
            "prompt": "Would you like `LICENSE.txt`",
            "help": (
                "If you plan to work on this project long term, you should include a"
                " license. By default, this will include the `Apache 2.0` license."
            ),
        },
    },
    "params": {
        "name": {
            "default": None,
            "prompt": "What is the name of your repo? E.g. MyPackage",
        },
        "package-name": {
            "default": None,
            "prompt": (
                "What is the name of your library when used in code? E.g. `import package_name`"
            ),
        },
        "organization": {
            "default": None,
            "prompt": (
                "What is the name of your organization?"
                " This is your usually your github name, e.g. `www.github.com/<organization>/<name>`"
            ),
        },
        "author": {
            "default": None,
            "prompt": "What is your name or authors names? E.g. LeetCookieEater and ToastySocks",
        },
        "email": {
            "default": "me@address.com",
            "prompt": "What email would you like associated with this repo? E.g. me@gmail.com",
        },
        "description": {
            "default": "No description given",
            "prompt": "What description would you like to give for your repo?",
        },
        "url": {
            "default": "https://www.automl.org",
            "prompt": "What url would you like to associate with this repo?",
        },
    },
}

# Anything not defined will be prompted
predefined_configs = {
    "student": {
        "testing": True,
        "docs": False,
        "packaging": False,
        "flake8": True,
        "pydocstyle": True,  # Have the editor shout for docs or I will
        "pre-commit": False,
        "citations": False,
        "license": False,
    },
    "publication": {
        "testing": True,
        "docs": True,
        "flake8": True,
        "pydocstyle": True,
        "black": True,
        "isort": True,
        "citations": True,
        "license": True,
    },
    "package": {
        "testing": True,
        "docs": True,
        "packaging": True,
        "flake8": True,
        "pydocstyle": True,
        "black": True,
        "isort": True,
        "pre-commit": True,
        "citations": True,
        "license": True,
    },
}


def path_replace(path: Path, part: Path, newpart: Path) -> Path:
    return Path(str(path).replace(str(part), str(newpart)))


def help(args: List[str]) -> str:
    err = f"Unrecognized command {' '.join(args)}."
    usage = f"Please use as `{args[0]} [config-kind]`"
    config_descriptions = [
        f"{name}\n\t" + "\n\t".join([f"{key} - {val}" for key, val in config.items()])
        for name, config in predefined_configs.items()
    ]
    return "\n".join(
        [
            err,
            "\n",
            usage,
            "\n",
            "[config-kind]",
            "\n",
            "\n\n".join(config_descriptions),
        ]
    )


def find_all(string: str, substring: str) -> List[int]:
    indices = []
    for i in range(len(string)):
        if string.startswith(substring, i):
            indices += [i]

    return indices


def find_and_replace(data: str, key: str, value: Union[str, bool]) -> str:
    """Find and replace"""
    # Should go in a function alone
    data = data.replace(f"<<{str(key)}>>", str(value))

    # Not needed if we use block_replace
    if not isinstance(value, bool):
        return data

    start_sequence = f"<<requires::{key}"
    end_sequence = f"endrequires::{key}>>"

    while True:
        starts = find_all(data, start_sequence)
        ends = find_all(data, end_sequence)

        if len(starts) != len(ends):
            raise SyntaxError(f"{start_sequence} and {end_sequence} are not balanced")

        if len(starts) == 0:
            break

        # Since the string is changing
        # We always use the first found and then search for more patterns
        start = starts[0]
        end = ends[0]

        # Simply remove the requires:: and endrequires::
        if value:

            # Remove all whitespaces and newlines after start
            content_start = start + len(start_sequence) + 1
            content_end = end

            beginning = data[:start]
            content = data[content_start:content_end]
            ending = data[end + len(end_sequence) :]

            # Remove spaces, tabs, newlines, ...
            content = content.lstrip()
            content = content.rstrip()

            data = beginning + content + ending

        # Remove the whole block
        else:
            beginning = data[:start]
            # beginning = beginning.rstrip()

            ending = data[end + len(end_sequence) :]
            ending = ending.lstrip()

            data = beginning + ending

    return data


def replace_templates(
    dir: Path,
    out: Path,
    params: Dict[str, str],
    features: Dict[str, bool],
) -> None:
    if not out.exists():
        out.mkdir()

    filepaths = map(Path, chain(dir.glob("**/*"), dir.glob("**/.*")))

    for filesrc in filepaths:

        # Skip directories, files will create them if needed
        if filesrc.is_dir():
            continue

        # Skip mypy cache if it tries to sneaks in
        if ".mypy_cache" in filesrc.parts:
            continue

        # Skip images
        extension = os.path.splitext(filesrc)[1]
        if extension in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"]:
            continue

        # Read in the file
        with filesrc.open("r", encoding="utf-8") as file:
            data = file.read()

        substitues: Dict[str, Union[str, bool]] = {**params, **features}
        for k, v in substitues.items():
            # Replace the target string
            data = find_and_replace(data, k, v)

        # Get the destination of the file
        filedst = path_replace(filesrc, TEMPLATE, TMPDIR)

        # Create the directory it should go in if doesn't exist
        if not filedst.parent.exists():
            filedst.parent.mkdir(parents=True)

        # Write the file out again
        with filedst.open("w", encoding="utf-8") as file:
            file.write(data)


def generate(params: Dict[str, str], features: Dict[str, bool]) -> None:
    replace_templates(TEMPLATE, TMPDIR, params, features)

    included_features = [key for key, val in features.items() if val is True]

    for filesrc, depends_on in files.items():
        if depends_on is None or any(set(depends_on) & set(included_features)):

            # Where the file has the templates filled in
            generated_template = path_replace(filesrc, TEMPLATE, TMPDIR)

            # Where it should go
            file_dst = path_replace(generated_template, TMPDIR, HERE)

            # If it's a directory, just make it if needed
            if file_dst.is_dir():
                if file_dst.exists():
                    continue
                else:
                    file_dst.mkdir()

            # If a path doesn't have its folder existing, create it
            if not file_dst.parent.exists():
                file_dst.parent.mkdir(parents=True)

            generated_template.replace(file_dst)

    # Rename the src folder
    src_folder = HERE / "src"
    src_newname = HERE / params["package-name"]
    src_folder.rename(src_newname)

    # Delete _template dir and _tmp
    for path in [TEMPLATE, TMPDIR]:
        rmtree(path)

    # Delete generate.py
    this_file = Path(__file__).resolve()
    this_file.unlink()


def get_params() -> Dict[str, str]:
    params = {}
    for key, desc in options["params"].items():
        prompt = desc["prompt"]
        default = desc.get("default", None)

        if default is not None:
            prompt += f"\tDefault: {default}"

        valid = False
        iters = 0
        while not valid:
            print(prompt)
            val = input("> ")

            if val not in ["", "\n", None]:
                valid = True

            elif default is not None:
                val = default
                valid = True

            elif iters <= 3:
                print("Please enter a value.\n")
                iters += 1

            else:
                print("Exiting")
                sys.exit(1)

        params[key] = val

    print("Your params: ")
    pprint(params, indent=4)

    print("\nConfirm our choices? (y/n)")
    val = input("> ")
    if val.lower() == "n":
        print("Exiting")
        sys.exit(1)
    else:

        return params


def get_features(config_kind: Optional[str]) -> Dict[str, bool]:
    if config_kind is None:
        features = {}
    else:
        features = predefined_configs[config_kind]

    feature_options = options["features"]
    missing_features = set(feature_options) - set(features)

    for key in missing_features:
        prompt = feature_options[key]["prompt"]
        help = feature_options[key]["help"]
        default = feature_options[key]["default"]

        print(f"-- {prompt}\n")
        print(f"{help}")

        default_str = "y" if default is True else "n"
        prompt += f" (y/n) [Default: {default_str} ]"

        print(prompt)
        value = input("> ")

        if value.lower() == "n":
            features[key] = False
        elif value.lower() == "y":
            features[key] = True
        else:
            features[key] = default

    print("Confirm? (y\\n)")
    pprint(features, indent=4)

    value = input("> ")
    if value.lower() == "n":
        print("Exiting setup")
        sys.exit(1)

    if set(features) & set(checkers):
        features["checkers"] = True

    if set(features) & set(formatters):
        features["formatters"] = True

    return features


if __name__ == "__main__":
    nargs = len(sys.argv)
    if nargs == 1:
        config_kind = None

    elif nargs == 2:
        config_kind = sys.argv[1]

    else:
        print(help(sys.argv))
        sys.exit(1)

    if config_kind is not None and config_kind not in predefined_configs:
        print(
            f"Unknown config {config_kind}."
            f"\nPlease provide a config from {list(predefined_configs)}"
        )
        sys.exit(1)

    params = get_params()
    features = get_features(config_kind)

    generate(params, features)
