import os
import pytest
from pyomo.common.fileutils import this_file_dir
from testbook import testbook
from omlt.dependencies import keras_available, onnx_available


#return testbook for given notebook
def openBook(folder, notebook_fname):
    os.chdir(os.path.join(this_file_dir(), '..', '..', 'docs', 'notebooks', folder))
    book = testbook(notebook_fname, execute=True, timeout=300)
    return book


#checks that the number of executed cells matches the expected
def run_notebook(tb, n_cells):
    assert tb.code_cells_executed == n_cells


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_relu_notebook():
    book = openBook('neuralnet', "auto-thermal-reformer-relu.ipynb")

    with book as tb:
        run_notebook(tb, 13)

        #check final values
        bypassFraction = tb.ref("pyo.value(m.reformer.inputs[0])")
        ngRatio = tb.ref("pyo.value(m.reformer.inputs[1])")
        h2Conc = tb.ref("pyo.value(m.reformer.outputs[h2_idx])")
        n2Conc = tb.ref("pyo.value(m.reformer.outputs[n2_idx])")

        assert bypassFraction == 0.1
        assert ngRatio == pytest.approx(1.12, abs=0.05)
        assert h2Conc == pytest.approx(0.33, abs=0.03)
        assert n2Conc == pytest.approx(0.34, abs=0.01)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_reformer():
    book = openBook('neuralnet', "auto-thermal-reformer.ipynb")

    with book as tb:
        run_notebook(tb, 13)

        #check final values
        bypassFraction = tb.ref("pyo.value(m.reformer.inputs[0])")
        ngRatio = tb.ref("pyo.value(m.reformer.inputs[1])")
        h2Conc = tb.ref("pyo.value(m.reformer.outputs[h2_idx])")
        n2Conc = tb.ref("pyo.value(m.reformer.outputs[n2_idx])")

        assert bypassFraction == pytest.approx(0.1, abs=0.001)
        assert ngRatio == pytest.approx(1.12, abs=0.03)
        assert h2Conc == pytest.approx(0.33, abs=0.01)
        assert n2Conc == pytest.approx(0.34, abs=0.01)


def test_build_network():
    book = openBook('neuralnet', "build_network.ipynb")

    with book as tb:
        run_notebook(tb, 37)

        #check for correct number of layers
        layers = tb.ref("m.neural_net.layer")
        assert len(layers) == 3


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

