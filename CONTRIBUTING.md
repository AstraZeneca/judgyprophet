# Contributing Guide

Thanks for thinking about contributing!

## Table of contents
* [Contributing process](#process)
* [Coding style](#style)

---
<a name="process"></a>
## Contributing Process

Please only make one change or add one feature per Pull Request (PR). Limit the scope of your PR so that it makes small, manageable size changes.

To help us integrate your changes, please follow our standard process:

Pre-condition:
* Make sure you have chatted to the judgyprophet team and let them know what you're planning, in case something similar is already in the works.


Dev Process:
1. Make a new issue (or use an existing one). You will need the issue number when you create a branch.
2. Clone this repo
3. Create a new branch. The branch name must include the issue number.
4. Install `judgyprophet` in development mode by running `poetry install`. This installs `pytest` for running tests, as well as the linters we use (`pylint` and `flake8`).
5. We are using pre-commit hooks to ensure code formatting and linting is done with every commit. Formatting is performed using the `black` package. To enable pre-commit settings, install the repo settings by running:
        ```pre-commit install```
6. Make your changes in small, logically grouped commits (this makes it easier to review changes)
    - document your code as you go
    - add integration tests for your code in the same style as the current tests (using test data that explicitly isolates the new feature)
7. Run tests and check they all pass using pytest: `pytest tests/`. When your changes are made, build the dockerfile and check the tests run there by using `docker build -t judgyprophet` followed by `docker run`.
8. Update the documentation with your changes
    - Documentation is created using [mkdocs](https://mkdocstrings.github.io/)
    - add new classes or modules to api docs
    - add new/changed functionality to the tutorials or quickstart
    - Check added documentation can be properly rendered by using `mkdocs serve` and checking rendered docs.
9. When finished, make a Pull Request (PR)
    - Describe change and clearly highlight any major or breaking changes
    - You are responsible for getting your PR merged so please assign reviewers and chase them to do the review
10. Adjust your PR based on any feedback
11. After approval, you are responsible for completing your PR

---

<a name="style"></a>
## Coding Style

### Docstrings
We use [Sphinx style docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)

### Type hinting
* Include type hinting in method signatures
* Include return types

### Linting
We loosely follow the pep8 style conventions, and use `flake8`'s bundled `pycodestyle` to check against these. We also use the `pylint` linter, aiming to fix any issues flagged as warning or above.

### Logging
Instead of print statements, use the python logging module to print to the user.

### File paths
Use pathlib for easier and safer file paths:

### Tests
Use pytest for running tests

##### Automated CI/CD tests
This is on the roadmap. It is not currently done.

---
