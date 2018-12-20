package com.willbooster.oop_excercise3.ml;

import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class XorTrainerByNeuralNetwork implements XorTrainer {
    public XorPredictorByNeuralNetwork train(MLDataSet trainingSet) {
        // ニューラルネットワークの構築（詳細は割愛）
        var network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 2));
        network.addLayer(new BasicLayer(new ActivationReLU(), true, 5));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();

        // 訓練データでニューラルネットワークをトレーニング
        var train = new ResilientPropagation(network, trainingSet);
        var epoch = 1;
        do {
            train.iteration();
            epoch++;
        } while (train.getError() > 0.01 && epoch < 100);
        train.finishTraining();

        return new XorPredictorByNeuralNetwork(network);
    }
}
