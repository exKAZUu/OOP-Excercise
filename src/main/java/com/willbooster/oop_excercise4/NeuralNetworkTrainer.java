package com.willbooster.oop_excercise4;

import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class NeuralNetworkTrainer implements Trainer {
    public NeuralNetworkPredictor train(MLDataSet trainingSet) {
        // create a neural network, without using a factory
        var network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 2));
        network.addLayer(new BasicLayer(new ActivationReLU(), true, 5));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();

        // train the neural network
        var train = new ResilientPropagation(network, trainingSet);

        var epoch = 1;

        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            epoch++;
        } while (train.getError() > 0.01);
        train.finishTraining();

        return new NeuralNetworkPredictor(network);
    }
}
