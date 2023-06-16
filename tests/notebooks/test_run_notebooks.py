import os
import pytest
from pyomo.common.fileutils import this_file_dir
from testbook import testbook
from omlt.dependencies import keras_available, onnx_available


#return testbook for given notebook
def openBook(folder, notebook_fname, **kwargs):
    execute = kwargs.get('execute', True)
    os.chdir(os.path.join(this_file_dir(), '..', '..', 'docs', 'notebooks', folder))
    book = testbook(notebook_fname, execute=execute, timeout=300)
    return book


#checks that the number of executed cells matches the expected
def check_cell_execution(tb, n_cells):
    assert tb.code_cells_executed == n_cells


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_relu_notebook():
    book = openBook('neuralnet', "auto-thermal-reformer-relu.ipynb")

    with book as tb:
        check_cell_execution(tb, 13)

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
        check_cell_execution(tb, 13)

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
        check_cell_execution(tb, 37)

        #check for correct number of layers
        net_layers = (tb.ref("list(net.layers)"))
        m_layers = tb.ref("m.neural_net.layer")
        assert len(net_layers) == 3
        assert len(m_layers) == 3

        #check eval function
        eval_ex = list(tb.ref("x"))
        assert eval_ex[0] == pytest.approx(2.15)


@pytest.mark.skipif(
    (not onnx_available) or (not keras_available),
    reason="onnx and keras needed for this notebook",
)
def test_import_network():
    book = openBook('neuralnet', "import_network.ipynb", execute=False)

    with book as tb:
        #inject cell that reads in loss and accuracy of keras model
        tb.inject("keras_loss, keras_accuracy = model.evaluate(X, Y)", before=25, run=False)
        tb.execute()

        #add one to true number because of injection
        check_cell_execution(tb, 17)

        #check input bounds
        input_bounds = tb.ref("input_bounds")
        assert input_bounds == [[0.0, 17.0],
                                [0.0, 199.0],
                                [0.0, 122.0],
                                [0.0, 99.0],
                                [0.0, 846.0],
                                [0.0, 67.1],
                                [0.078, 2.42],
                                [21.0, 81.0]]
        
        #checking accuracy and loss of keras model
        model = tb.ref('model')
        print(model)
        keras_loss, keras_accuracy = tb.ref('keras_loss'), tb.ref("keras_accuracy")
        assert keras_loss == pytest.approx(5.4, abs=4.8)
        assert keras_accuracy == pytest.approx(0.45, abs=0.21)

        #checking loss of pytorch model
        pytorch_loss = tb.ref("loss.item()")
        assert pytorch_loss == pytest.approx(0.25, abs=0.1)

        #checking the model that was imported
        imported_input_bounds = tb.ref('network_definition.scaled_input_bounds')
        assert imported_input_bounds == {'0': [0.0, 17.0],
                                         '1': [0.0, 199.0],
                                         '2': [0.0, 122.0],
                                         '3': [0.0, 99.0],
                                         '4': [0.0, 846.0],
                                         '5': [0.0, 67.1],
                                         '6': [0.078, 2.42],
                                         '7': [21.0, 81.0]}

        #checking the imported layers
        tb.inject("""
                    activations = ['linear', 'relu', 'relu', 'linear']
                    for layer_id, layer in enumerate(network_definition.layers):
                        assert activations[layer_id] == layer.activation
                """)


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_convolutional():
    book = openBook('neuralnet', "mnist_example_convolutional.ipynb")

    with book as tb:
        check_cell_execution(tb, 13)


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_dense():
    book = openBook('neuralnet', "mnist_example_dense.ipynb")

    with book as tb:
        check_cell_execution(tb, 13)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_neural_network_formulations():
    book = openBook('neuralnet', "neural_network_formulations.ipynb")

    with book as tb:
        check_cell_execution(tb, 21)

@pytest.mark.skipif(not onnx_available, reason='onnx needed for this notebook')
def test_bo_with_trees():
    book = openBook('', "bo_with_trees.ipynb")

    with book as tb:
        check_cell_execution(tb, 10)

