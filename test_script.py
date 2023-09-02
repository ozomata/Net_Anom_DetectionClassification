import unittest
from CNNBiLSTM_Classifier_NSLKDD import one_hot, cnnbilstm 

def test_cnnbilstm():
    # Check if the model is created correctly
    model = cnnbilstm()
    assert isinstance(model, Sequential)
    assert len(model.layers) == 11

    # Check if the model summary is correct
    summary = model.summary()
    assert "Convolution1D" in summary
    assert "MaxPooling1D" in summary
    assert "Bidirectional" in summary
    assert "Dense" in summary

    # Check if the model can be trained
    model.fit(X_train, y_train)

    # Check if the model can be evaluated
    model.evaluate(X_test, y_test)

if __name__ == "__main__":
    unittest.main()