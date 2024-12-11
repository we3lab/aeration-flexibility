*******************************
{{ cookiecutter.project_name }}
*******************************

.. image::
   https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/workflows/Build%20Main/badge.svg
   :height: 30
   :target: https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/actions
   :alt: Build Status

.. image::
   https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/workflows/Documentation/badge.svg
   :height: 30
   :target: https://we3lab.github.io/{{ cookiecutter.project_slug }}
   :alt: Documentation

.. image::
   https://codecov.io/gh/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/branch/main/graph/badge.svg
   :height: 30
   :target: https://codecov.io/gh/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}
   :alt: Code Coverage

{{ cookiecutter.project_short_description }}

Features
========
- Store values and retain the prior value in memory
- ... some other functionality

Quick Start
===========
```python
from {{ cookiecutter.project_slug }} import Example

a = Example()
a.get_value()  # 10
```

Installation
============
- **Stable Release:** `pip install {{ cookiecutter.project_slug }}`
- **Development Head:** `pip install git+https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}.git`

Documentation
=============
For full package documentation please visit [{{ cookiecutter.github_username }}.github.io/{{ cookiecutter.project_slug }}](https://{{ cookiecutter.github_username }}.github.io/{{ cookiecutter.project_slug }}).

Development
===========

See [CONTRIBUTING.rst](CONTRIBUTING.rst) for information related to developing the code.

Useful Commands
===============

1. ``pip install -e .``

  This will install your package in editable mode.

2. ``pytest {{ cookiecutter.project_slug }}/tests --cov={{ cookiecutter.project_slug }} --cov-report=html``

  Produces an HTML test coverage report for the entire project which can
  be found at ``htmlcov/index.html``.

3. ``docs/make html``

  This will generate an HTML version of the documentation which can be found
  at ``_build/html/index.html``.

4. ``flake8 {{ cookiecutter.project_slug }} --count --verbose --show-source --statistics``

  This will lint the code and share all the style errors it finds.

5. ``black {{ cookiecutter.project_slug }}``

  This will reformat the code according to strict style guidelines.

Legal Documents
===============
- `LICENSE <https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/blob/main/LICENSE/>`_
