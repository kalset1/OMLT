import os

import pytest
from pyomo.common.fileutils import this_file_dir
from testbook import testbook

from omlt.dependencies import keras_available, onnx_available


# TODO: These will be replaced with stronger tests using testbook soon

def openBook(folder, notebook_fname):
    cwd = os.getcwd()
    os.chdir(os.path.join(this_file_dir(), '..', '..', 'docs', 'notebooks', folder))
    book = testbook(notebook_fname, execute=True, timeout=300)
    return book

def run_notebook(tb, n_cells):
    assert tb.code_cells_executed == n_cells

@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_relu_notebook():
    book = openBook('neuralnet', "auto-thermal-reformer-relu.ipynb")
    with book as tb:
        run_notebook(tb, 13)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_reformer():
    book = openBook('neuralnet', "auto-thermal-reformer.ipynb")
    with book as tb:
        run_notebook(tb, 13)


def test_build_network():
    book = openBook('neuralnet', "build_network.ipynb")
    with book as tb:
        run_notebook(tb, 37)


@pytest.mark.skipif(
    (not onnx_available) or (not keras_available),
    reason="onnx and keras needed for this notebook",
)
def test_import_network():
    book = openBook('neuralnet', "import_network.ipynb")
    with book as tb:
        run_notebook(tb, 16)


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_convolutional():
    book = openBook('neuralnet', "mnist_example_convolutional.ipynb")
    with book as tb:
        run_notebook(tb, 13)


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_dense():
    book = openBook('neuralnet', "mnist_example_dense.ipynb")
    with book as tb:
        run_notebook(tb, 13)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_neural_network_formulations():
    book = openBook('neuralnet', "neural_network_formulations.ipynb")
    with book as tb:
        run_notebook(tb, 21)

@pytest.mark.skipif(not onnx_available, reason='onnx needed for this notebook')
def test_bo_with_trees():
    book = openBook('', "bo_with_trees.ipynb")
    with book as tb:
        run_notebook(tb, 10)

