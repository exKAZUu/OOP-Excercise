package com.willbooster.oop_excercise_final.ml;

import com.willbooster.oop_excercise_final.XorPredictor;
import org.encog.ml.data.MLData;
import org.encog.neural.networks.BasicNetwork;

public class NeuralNetworkXorPredictor implements XorPredictor {
    private BasicNetwork network;

    public NeuralNetworkXorPredictor(BasicNetwork network) {
        this.network = network;
    }

    public MLData predict(MLData input) {
        MLData data = this.network.compute(input);
        data.setData(0, data.getData(0) >= 0.0 ? 0.0 : 1.0);
        return data;
    }
}
