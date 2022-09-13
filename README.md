# OnsagerNet

This is a work in progress for the implementation of OnsagerNet and applications.

## Installation

This project uses `python 3.9`. Set up the project for development using the following steps:

1. Create a virtual environment
    ```bash
    $virtualenv -p python3.9 ~/.virtualenvs/onsager
    ```
2. Activate the environment
    ```bash
    $source ~/.virtualenvs/onsager/bin/activate
    ```
3. Install requirements
    ```bash
    $pip install -r requirements.txt
    ```
4. Perform editable install for development
    ```bash
    $pip install -e .
    ```

## Generate Documentation

Generate sphinx documentation:

1. Go to docs directory
    ```shell
    $cd docs
    ```
2. Make the apidoc
    ```shell
    $make rst
    ```
3. Generate html
    ```shell
    $make html
    ```
4. You can view the html docs on MacOs by
    ```shell
    $make view
    ```
    Otherwise, open `docs/build/html/index.html` with any browser.

## Quickstart (development)

Look at [examples](./examples).
