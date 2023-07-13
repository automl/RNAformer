# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

NAME := <<name>>
PACKAGE_NAME := <<package-name>>

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
<<requires::packaging DIST := dist endrequires::packaging>>
<<requires::docs
DOCDIR := docs
INDEX_HTML := "file://${DIR}/docs/build/html/index.html"
EXAMPLES_DIR := examples
endrequires::docs>>
<<requires::testing TESTS_DIR := tests endrequires::testing>>

.PHONY: help install-dev <<requires::checkers check endrequires::checkers>> <<requires::formatters format endrequires::formatters>> <<requires::pre-commit pre-commit endrequires::pre-commit>> clean <<requires::docs docs clean-doc examples endrequires::docs>> <<requires::packaging clean-build build publish endrequires::packaging>> <<requires::testing test endrequires::testing>>

help:
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	<<requires::checkers @echo "* check            to check the source code for issues" endrequires::checkers>>
	<<requires::formatters @echo "* format           to format the code with black and isort" endrequires::formatters>>
	<<requires::pre-commit @echo "* pre-commit       to run the pre-commit check" endrequires::pre-commit>>
	<<requires::packaging @echo "* build            to build a dist" endrequires::packaging>>
	<<requires::docs @echo "* docs             to generate and view the html files, checks links" endrequires::docs>>
	<<requires::docs @echo "* examples         to run and generate the examples" endrequires::docs>>
	<<requires::packaging @echo "* publish          to help publish the current branch to pypi" endrequires::packaging>>
	<<requires::testing @echo "* test             to run the tests" endrequires::testing>>

PYTHON ?= python
<<requires::testing PYTEST ?= python -m pytest endrequires::testing>>
PIP ?= python -m pip
MAKE ?= make
<<requires::black BLACK ?= black endrequires::black>>
<<requires::isort ISORT ?= isort endrequires::isort>>
<<requires::pydocstyle PYDOCSTYLE ?= pydocstyle endrequires::pydocstyle>>
<<requires::mypy MYPY ?= mypy endrequires::mypy>>
<<requires::pre-commit PRECOMMIT ?= pre-commit endrequires::pre-commit>>
<<requires::flake8 FLAKE8 ?= flake8 endrequires::flake8>>

install-dev:
	$(PIP) install -e ".[dev]"
	<<requires::pre-commit pre-commit install endrequires::pre-commit>>

<<requires::black
check-black:
	$(BLACK) ${SOURCE_DIR} --check || :
	<<requires::docs $(BLACK) ${EXAMPLES_DIR} --check || : endrequires::docs>>
	<<requires::testing $(BLACK) ${TESTS_DIR} --check || : endrequires::testing>>
endrequires::black>>

<<requires::isort
check-isort:
	$(ISORT) ${SOURCE_DIR} --check || :
	<<requires::testing $(ISORT) ${TESTS_DIR} --check || : endrequires::testing>>
endrequires::isort>>

<<requires::pydocstyle
check-pydocstyle:
	$(PYDOCSTYLE) ${SOURCE_DIR} || :
endrequires::pydocstyle>>

<<requires::mypy
check-mypy:
	$(MYPY) ${SOURCE_DIR} || :
endrequires::mypy>>

<<requires::flake8
check-flake8:
	$(FLAKE8) ${SOURCE_DIR} || :
	$(FLAKE8) ${TESTS_DIR} || :
endrequires::flake8>>

<<requires::checkers check: <<requires::black check-black endrequires::black>> <<requires::isort check-isort endrequires::isort>> <<requires::mypy check-mypy endrequires::mypy>> <<requires::flake8 check-flake8 endrequires::flake8>> <<requires::pydocstyle check-pydocstyle endrequires::pydocstyle>> endrequires::checkers>>

<<requires::pre-commit
pre-commit:
	$(PRECOMMIT) run --all-files
endrequires::pre-commit>>

<<requires::black
format-black:
	$(BLACK) ${SOURCE_DIR}
	<<requires::testing $(BLACK) ${TESTS_DIR} endrequires::testing>>
	<<requires::docs $(BLACK) ${EXAMPLES_DIR} endrequires::docs>>
endrequires::black>>

<<requires::isort
format-isort:
	$(ISORT) ${SOURCE_DIR}
	<<requires::testing $(ISORT) ${TESTS_DIR} endrequires::testing>>
endrequires::isort>>

<<requires::formatters
format: <<requires::black format-black endrequires::black>> <<requires::isort format-isort endrequires::isort>>
endrequires::formatters>>

<<requires::testing
test:
	$(PYTEST) ${TESTS_DIR}
endrequires::testing>>

<<requires::docs
clean-doc:
	$(MAKE) -C ${DOCDIR} clean

docs:
	$(MAKE) -C ${DOCDIR} docs
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

examples:
	$(MAKE) -C ${DOCDIR} examples
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}
endrequires::docs>>

<<requires::packaging
clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py sdist

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean build
	read -p "Did you update the version number?"
	
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uplaoded distribution into"
	@echo "* Run the following:"
	@echo
	@echo "        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${NAME}"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo
	@echo "        python -c 'import ${PACKAGE_NAME}'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo
	@echo "    python -m twine upload dist/*"
endrequires::packaging>>

# Clean up any builds in ./dist as well as doc, if present
clean: <<requires::packaging clean-build endrequires::packaging>> <<requires::docs clean-doc endrequires::docs>>
