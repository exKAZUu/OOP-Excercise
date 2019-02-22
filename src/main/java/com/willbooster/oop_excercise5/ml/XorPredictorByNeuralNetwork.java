package com.willbooster.oop_excercise5.ml;

import com.willbooster.oop_excercise5.XorPredictor;
import org.encog.ml.data.MLData;
import org.encog.neural.networks.BasicNetwork;

public class XorPredictorByNeuralNetwork extends XorPredictor {
    private BasicNetwork network;

    public XorPredictorByNeuralNetwork(BasicNetwork network) {
        this.network = network;
    }

    public int predict(MLData input) {
        validateInput(input);
        MLData data = this.network.compute(input);
        return data.getData(0) <= 0.5 ? 0 : 1;
    }
}
