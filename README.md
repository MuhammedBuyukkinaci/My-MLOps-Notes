# My-MLOps-Notes
Listing my MLOps learnings

This repository is containing my notes from [this Udemy course](https://www.udemy.com/course/complete-mlops-bootcamp-from-zero-to-hero-in-python-2022/).

1) 85 percent of trained ML model don't reach production and 55 % of companies don't deploy a single model.

2) An ideal ML life cycle

![ml_life_cycle](./images/001.png)

3) Researches show that companies using UI increased their profit margin by 3% to 15%.

4) DevOps applied to Machine Learning is known as MLOps. Model creation must be scalable, collaborative and reproducible. The principles, tools and techniques that make models scalable, collaborative and reproducible are known as MLOps.

5) MLOps process:

![mlops_process](./images/002.png)

6) DevOps applied to Machine Learning is known as MLOps. DevOps applied to Data is known as DataOps.

7) Roles in MLOps

![roles](./images/003.png)

8) Challenges addressed by MLOps

- Data and Artifact versioning

- Model Tracking: Degradition of performance due to data drift.

- Feature Generation: MLOPS allows to reuse methods

9) Parts of MLOPS

![part1](./images/004.png)

![part2](./images/005.png)

![part_all](./images/006.png)

10) MLOps Tools

![tools](./images/007.png)

11) Some data labelling tools:

- [v7labs](https://www.v7labs.com/pricing)

- [labelbox](https://labelbox.com/pricing/)

12) Some Feature Engineering Tools:

- [feast](https://github.com/feast-dev/feast)

- [featuretools](https://github.com/alteryx/featuretools)

- [tsfresh](https://github.com/blue-yonder/tsfresh)

13) Some Hyperparameter Optimization Tools:

- [Optuna](https://optuna.org/)

- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)

14) Fast API can be used in serving ML model.

15) Streamlit is useful for POC.

16) MLOps stages:

![ml_ops_stages](./images/008.png)

17) Some tools to use

![ml_ops_stages](./images/009.png)

18) Structuring ML projects in one of 3 ways.

![structuring](./images/010.png)

## Cookiecutter

19) Cookiecutter is a tool to structure our ML projects and folders.

```runall.sh
pip install cookiecutter

cookiecutter https://github.com/khuyentran1401/data-science-template
```

![cookiecutter](./images/011.png)

## Poetry

20) [Poetry](https://python-poetry.org/) allows us to manage dependencies and versions. Poetry is an alternative to pip.

    - Poetry separates main dependencies and sub dependencies into two separate files. Whereas,pip stores all dependencies in a single file(requirements.txt).
    - Poetry creates readable dependency files.
    - Poetry removes all sub dependencies when removing a library
    - Poetry avoids installing new libraries in conflict with existing libraries.
    - Poetry packages project with few lines of code.
    - All the dependencies of he project are specified in pyproject.toml

```poetry.sh
# TO install poetry on your machine(for linux and mac)
curl -sSL https://install.python-poetry.org | python3 -


# To generate a project
poetry new <project_name>

# To install dependencies
poetry install

# To add a new pypi library
poetry add <library_name>

# To delete a library
poetry remove <library_name>

# To show installed libraries
poetry show

# To show sub dependencies
poetry show --tree

# Link our existing environment(venv, conda etc) to poetry
poetry env use /path/to/python
```

21) [Hydra](https://hydra.cc/docs/intro/) manages configuration files. It makes project management easier.

    - Configuration information shouldn't be mixed with main code.
    - It is easier to modify things in a configuration file.
    - YAML is a common language for a configuration file.
    - An example config file and its usage via hydra
    - We can modify hydra parameters via CLI without modifying config file.

    ![Hydra](./images/014.png)

    ![Hydra](./images/015.png)

    - Hydro logging is super useful.

    - To use hydra, we must add config as an argument to a function.

```
import hydra
from pipeline2 import pipeline2

@hydra.main(config_name = 'preprocessing')
def run_training(config):

    match_pipe = pipeline2(config)


```

![Hydra](./images/013.png)

![Hydra](./images/017.png)

22) [Pre-commit](https://pre-commit.com/) plugins: It automates code review and formatting. In order to install them, use `pip install pre-commit`. After installing `pre-commit`, fill out `.pre-commit-config.yaml` and run `pre-commit install` to install it. Then, some checks are run before committing to local repository. Commit will not be done until the problem got solved. `--no-verify` is flag that can be appended to git commit. It doesn'T force you to correct the mistakes detected by pre-commit.

![precommit](./images/018.png)

    - Formatter: black
    - PEP8 Checker: flake8
    - Sort imports: isort
    - Check for docstrings: interrogate

![precommit](./images/019.png)

23) Black and Flake8

```run.sh
# pip install black
black file_name_01.py

# pip install flake8
flake8 temp.py
```

24) isort and iterrogate

    - correct isort:
    ![isort](./images/020.png)

```isort_usage.py
#pip install isort
isort file_name.py
#pip install interrogate
interrogate -vv file_name.py

```

25) [DVC](https://dvc.org/) is used for version control of model training data.

26) [pdoc](https://github.com/mitmproxy/pdoc) is used to automatically create documentation for projects.

```install.sh
pip install pdoc3

pdoc --http localhost:8080 temp.py

```

27) Makefile creates short and readable commands for configuration tasks. We can use Makefile to automate tasks such as setting up the environment.

![Makefile](./images/012.png)

28) A solution design is available at [here](https://github.com/ttzt/catalog_of_requirements_for_ai_products)

29) MLOps stages:

![mlops_stages](./images/021.png)

30) What AutoML does:

![mlops_stages](./images/022.png)

31) PyCaret is an open source, low code ML library. It has been developed in Python and reduce the time needed to create a model to minutes.

![pycaret](./images/023.png)

32) [PyCaret](https://pycaret.org/) incorporates these libraries:

![pycaret](./images/024.png)

33) [Pandas Profiling](https://github.com/ydataai/pandas-profiling) is allowing us to develop an exhaustive analysis of data.

![pandas_profiling](./images/025.png)

34) An example of PyCaret setup function:

![pycaret](./images/026.png)

35) Tukey-Anscomble Plot && Normal QQ Plot

![plot1](./images/027.png)

36) Scale-Location Plot && Residuals & Leverage

![plot2](./images/028.png)

37) MLOps Tracking Server and Model Registry

![plot2](./images/029.png)

38) MLFlow UI for different runs

![plot2](./images/030.png)

39) Different Components of MLFlow

40) We can log parameters, metrics and models in MLFlow.

```mlflow_demo.py

import mlflow
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse

alpha = 0.5

def rmse_compute(true,preds):
    pass

X_train = None
y_train = None

X_test = None
y_test = None

with mlflow.start_run():

    lr = LogisticRegression(alpha = alpha)
    lr.fit(X_train,y_train)
    y_test_preds = lr.predict(X_test)
    rmse = rmse_compute(y_test,y_test_preds)
    mlflow.log_param('alpha',alpha)
    mlflow.log_metric('rmse',rmse)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != 'file':
        mlflow.sklearn.log_model(lr,'model',registered_monel_name = 'ElasticNetWineModel')
    else:
        mlflow.sklearn.log_model(lr,'model')


```

41) We can register models into MLFlow via PyCaret.

```
#pass log_experiment = True, experiment_name = 'diamond'

s = setup(data, target = 'Precio', transform_target = True, log_experiment = True, experiment_name = 'diamond')

```






