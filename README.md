# AutoML Template
A template that provides all the tools to ensure the same project setup across all AutoML packages.

You can follow [githubs docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) for copying the template then follow the commands below to generate the template.

1. On the repo [`automl_template`](https://github.com/automl/automl_template) hit **Use this template**.
2. Select yourself, the owner of the new repository.
3. Give your repository a name.
4. Make it public/private. If making it private, you'll have automated [workflow](#workflows) limits. By default these
   are disabled but you can navigate to `.github/workflows` and uncomment everything under `on:` to enable them.
5. Select **Include all branches** if you want published docs. May as well even if you don'tn need docs yet!
6. Click **Create repository from template**

Once you've done those steps, you'll want to download your repo, generate the template
and push up your generated template to github.

```bash
git clone git@github.com:user/myrepo.git

# Move into the cloned repo
cd myrepo

# Generate the repo , optionally with a predefined config
python generate.py # student publication package

# ... You'll be asked things about your repo, fill them in and confirm

# Add the new repo contents
git add .

# Add a commit with whatever message you like
git commit -m "*mario voice* Wahoooo"
git push

# ... and you're good to go!
```

Some next steps to do would be to create a virtual environment and use that and then install your repo
install your repo with the `[dev]` requirements that were generated for you. You can find these
in `setup.py` if you're curious!

```
# Create a virtual env with pythons built in virtualenv
python -m venv myenv
source ./myenv/bin/activate

# ... or using conda
conda create -n myenv python=3.8
conda activate myenv

# Install the repo
make install-dev

# ... or manually
pip install -e ".[dev]"

# If doing manually and you activated the `pre-commit` feature, you'll need to run this too
pre-commit install
```

You can check out the `Makefile` where we put some useful commands for your new repo!
```
make help
```

Lastly, you can [configure your editor](#configuring-your-editor) to use all these nice new tools :)

## Use cases
We support 3 main use cases, which you kind find their features in the [table overview](#config-table):
* `student` - Includes some checkers with support for testing.
* `publication` - A repo for a publication, same as student but with some extras included: doc building, formatters, a citation file and a license.
* `package` - A repo that's planned to have multiple contributors and published to PyPi. Same as publication but with pre-commit to help with code and some utility to help with publishing.

We set some sensible defaults but you'll still be asked about some extra [features](#features) you can optionally
include if you like.
You can get a [table overview](#config-table) of what's included in or view each individual tool [here](#features).

## Parameters
You'll be asked about these any time you run `generate.py` with a brief description.

* **name** - The name of your github repo when you created the template
* **package-name** - The name of your package when doing `import my_package_name`
* **author** - Your name (and collaborators names)
* **organization** - The name of your organization. This is your github name `www.github.com/<organization>/my_repo_name`
* **email** (_optional_) - An email address you would like associated with your repo
* **description** (_optional_) - A description of your repo
* **url** (_optional_) - A url you would like associated with your repo

You can leave any (_optional_) things blank if not relevant.

## Config Table

| **feature**               | `student`          | `publication`      | `package`          | _empty_         |
| -----------               | ---------          | -------------      | ---------          | -------         |
| [testing](#testing)       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :grey_question: |
| [flake8](#flake8)         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :grey_question: |
| [pydocstyle](#pydocstyle) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :grey_question: |
| [docs](#docs)             | :x:                | :heavy_check_mark: | :heavy_check_mark: | :grey_question: |
| [black](#black)           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :grey_question: |
| [isort](#isort)           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :grey_question: |
| [citations](#citations)   | :x:                | :heavy_check_mark: | :heavy_check_mark: | :grey_question: |
| [license](#license)       | :x:                | :heavy_check_mark: | :heavy_check_mark: | :grey_question: |
| [packaging](#packaging)   | :x:                | :x:                | :heavy_check_mark: | :grey_question: |
| [pre-commit](#pre-commit) | :x:                | :x:                | :heavy_check_mark: | :grey_question: |
| [mypy](#mypy)             | :grey_question:    | :grey_question:    | :grey_question:    | :grey_question: |

## Features

#### testing
For testing, we include [`pytest`](https://docs.pytest.org/en/6.2.x/contents.html) and also provide a workflow file [`pytest.yml`](#pytest.yml). The tests can be run locally with `pytest` or `make test` while every push will
also activate the workflow so you get automated tests on github.
* configuration - `pyproject.toml`
* run - `make test` or `pytest`
* workflow - [`pytest.yml`](#pytest.yml)

#### flake8
This includes a code checking tool [`flake8`](https://flake8.pycqa.org/en/latest/) which checks for common programmer errors and that your code complies with [`pep8`](https://www.python.org/dev/peps/pep-0008/) standards.
* configuration `.flake8`
* run - `make check` or `flake8 <path>`

#### pydocstyle
This includes [`pydocstyle`](http://www.pydocstyle.org/en/stable/) to your code to make sure you include some documentation so that anyone reading your code will have a much easier time.
* configuration - `pyproject.toml`.
* run - `make check` or `pydocstyle <path>`

#### docs
This includes tooling around generating [`sphinx`](https://www.sphinx-doc.org/en/master/) documentation and provides a nice default theme from [`automl_sphinx_theme`](https://github.com/automl/automl_sphinx_theme).

This feature will enable **github pages** and automatically generate your docs which you can view
at `https://organization.github.io/myreponame/main/`. An example of how that looks is [here](https://automl.github.io/automl_sphinx_theme/main/). This also includes a place to put [examples](https://automl.github.io/automl_sphinx_theme/main/examples/index.html) which will be run and produce the output into your docs. This is done
automatically with a workflow [`docs.yml`](#docs.yml). You'll want to add this to your repos home page on the right for easy access ;)

* configuration - `docs/conf.py`
* run - `make docs` or `make examples` inside `docs`

Your logo and favicon are shown if the files `docs/images/logo.png` or `docs/images/favicon.ico` can be found, respectively.

#### black
This feature will include a formatter called [`black`](https://black.readthedocs.io/en/stable/) which automatically formats your code to a stricter version of [`pep8`](https://www.python.org/dev/peps/pep-0008/) standards. This is useful for repos which will have multiple contributors so reviews can focus on the code changes and not stylistic changes.
* configuration - `pyproject.toml`
* run - `make format` or `black <path>`

#### isort
This feature will include the [`isort`](https://pycqa.github.io/isort/) formatter which automatically sorts your imports. For the same reasons as `black`, this means imports are done in a uniform manner for all contributors.
* configuration - `pyproject.toml`
* run - `make format` or `isort <path>`

#### citations
This includes a `CITATION.cff` which allows users to automatically grab a citation from your github repos homepage. It also includes a `citation_cff.yml` workflow to automatically validate it anytime you update it. You can find more [here](https://citation-file-format.github.io/) and more detailed information [here](https://github.com/citation-file-format/citation-file-format/blob/main/README.md).
* workflows - `citations_cff.yml`

#### license
If you plan to have others use your code beyond trying things out, you'll want to include a license. By default we only include the [Apache2.0 license](https://www.apache.org/licenses/LICENSE-2.0) but you can change this if you require.

#### pre-commit
This feature enables [`pre-commit`](https://pre-commit.com/), a tool that integrates with `git` so that every time you try to commit with `git commit` or in your editor, such as vscode. This will automatically run any of the checkers you have installed so that it will prompt you to fix errors before you push the code up to the repository.
* configuration - `pre-commit-config.yml`
* run - `make pre-commit` or `pre-commit run --all-files`
* workflows - `pre-commit.yml`

#### mypy
The feature we opted to always be optional is `mypy`. This is a static type checker that works with python's [`typing`](https://docs.python.org/3/library/typing.html) module. This can include help catch quite a few errors before any code is even run, similar to how `C` or `Java` code won't compile if there's type errors, but allows for gradual typing, allowing you to "type" the parts of the code that matter to you.

If you're **not** familiar with pythons typing or typing in general, we recommend you keep this feature **off**
* configuration - `pyproject.toml`
* run - `make check` or `mypy <path>`

### Workflows
Workflows are what github uses to automatically run certain things anytime there's a change in the repository.
You probably don't need to mess with these files too much but you can find them in the `.github/workflows` folder. Github's documentation for this is [here](https://docs.github.com/en/actions/learn-github-actions)

By default these are disabled but you can navigate to `.github/workflows` and uncomment everything under `on:` to enable them.

#### `citations_cff.yml`
This workflow activates whenever you change your `CITATION.cff` file and validates it.
* path - `.github/workflows/citatin_cff.yml`

#### `pre-commit.yml`
This workflow will run all the checkers and formatters against your code and fail if they don't pass.
* path - `.github/workflows/pre-commit.yml`

#### `pytest.yml`
This workflow will run any tests you have against your code and fail if any of the tests fail
* path `.github/workflows/pytest.yml`

#### `docs.yml`
This workflow will build your documentation and fail if it can't build the docs. If the changes
were to the main branch, it will push the new documentation to github pages so it can be viewed publicly
* path - `.github/workflows/docs.yml`

## Configuring your editor

#### vscode
Just search for the "python" plugin and install it. Your editor should then pick up on all the configuration!

#### Pycharm
Help wanted, please raise an issue with how to do this :)

#### Other
If you use other editors and know how to configure them for these tools, raise an issue [here](https://github.com/automl/automl_template/issues) and let us know!
