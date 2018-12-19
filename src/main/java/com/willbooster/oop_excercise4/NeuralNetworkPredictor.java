package com.willbooster.oop_excercise4;

import org.encog.ml.data.MLData;
import org.encog.neural.networks.BasicNetwork;

public class NeuralNetworkPredictor implements Predictor {
    private BasicNetwork network;

    public NeuralNetworkPredictor(BasicNetwork network) {
        this.network = network;
    }

    public MLData predict(MLData input) {
        return this.network.compute(input);
    }
}
