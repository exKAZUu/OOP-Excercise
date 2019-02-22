package com.willbooster.answer_oop_excercise6.ml;

import com.willbooster.answer_oop_excercise6.XorPredictor;
import org.encog.ml.data.MLData;
import org.encog.neural.networks.BasicNetwork;

public class XorPredictorByNeuralNetwork extends XorPredictor {
    private BasicNetwork network;

    public XorPredictorByNeuralNetwork(BasicNetwork network) {
        this.network = network;
    }

    public int predictBody(MLData input) {
        MLData data = this.network.compute(input);
        return data.getData(0) <= 0.5 ? 0 : 1;
    }
}
