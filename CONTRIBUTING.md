# Contributing to DepthAI Nodes

It outlines our workflow and standards for contributing to this project.

## Table of Contents

- [Developing parser](#developing-parser)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Documentation](#documentation)
  - [Editor Support](#editor-support)
- [Making and Reviewing Changes](#making-and-reviewing-changes)

## Developing parser

Parser should be developed so that it is consistent with other parsers. Check out other parsers to see the required structure. Additionally, pay attention to the naming of the parser's attributes. Check out [NN Archive Parameters](docs/nn_archive_parameters.md).

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency:

1. Install pre-commit (see [pre-commit.com](https://pre-commit.com/#install)).
1. Clone the repository and run `pre-commit install` in the root directory.
1. The pre-commit hook will now run automatically on `git commit`.
   - If the hook fails, it will print an error message and abort the commit.
   - It will also modify the files in-place to fix any issues it can.

## Documentation

We use the [Epytext](https://epydoc.sourceforge.net/epytext.html) markup language for documentation.
To verify that your documentation is formatted correctly, follow these steps:

1. Download [`get-docs.py`](https://github.com/luxonis/python-api-analyzer-to-json/blob/main/gen-docs.py) script
1. Run `python3 get-docs.py luxonis_ml` in the root directory.
   - If the script runs successfully and produces `docs.json` file, your documentation is formatted correctly.
   - **NOTE:** If the script fails, it might not give the specific error message. In that case, you can run
     the script for each file individually until you find the one that is causing the error.

### Editor Support

- **PyCharm** - built in support for generating `epytext` docstrings
- **Visual Studie Code** - [AI Docify](https://marketplace.visualstudio.com/items?itemName=AIC.docify) extension offers support for `epytext`
- **NeoVim** - [vim-python-docstring](https://github.com/pixelneo/vim-python-docstring) supports `epytext` style

## Making and Reviewing Changes

1. Make changes in a new branch.
1. Test your changes locally.
1. Commit (pre-commit hook will run).
1. Push to your branch and create a pull request. Always request a review from:
   - [Matija Teršek](https://github.com/tersekmatija)
   - [Jakob Mraz](https://github.com/jkbmrz)
   - [Jaša Kerec](https://github.com/kkeroo)
   - [Klemen Škrlj](https://github.com/klemen1999)
1. Any other relevant team members can be added as reviewers as well.
1. The team will review and merge your PR.
