package com.willbooster.answer_oop_excercise1.ml;

import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class XorByNeuralNetwork implements XorTrainer, XorPredictor {
    private BasicNetwork network;

    public void train(MLDataSet trainingSet) {
        // ニューラルネットワークの構築（詳細は割愛）
        this.network = new BasicNetwork();
        this.network.addLayer(new BasicLayer(null, true, 2));
        this.network.addLayer(new BasicLayer(new ActivationReLU(), true, 5));
        this.network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        this.network.getStructure().finalizeStructure();
        this.network.reset();

        // 訓練データでニューラルネットワークをトレーニング
        var train = new ResilientPropagation(this.network, trainingSet);
        var epoch = 1;
        do {
            train.iteration();
            epoch++;
        } while (train.getError() > 0.01 && epoch < 100);
        train.finishTraining();
    }

    public int predict(MLData input) {
        MLData data = this.network.compute(input);
        return data.getData(0) <= 0.5 ? 0 : 1;
    }
}
