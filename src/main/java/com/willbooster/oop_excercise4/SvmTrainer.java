package com.willbooster.oop_excercise4;

import org.encog.ml.data.MLDataSet;
import org.encog.ml.svm.SVM;
import org.encog.ml.svm.training.SVMSearchTrain;

public class SvmTrainer implements Trainer {
    public SvmPredictor train(MLDataSet trainingSet) {
        var svm = new SVM(2, false);

        // train the neural network
        var train = new SVMSearchTrain(svm, trainingSet);

        var epoch = 1;
        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            epoch++;
        } while (train.getError() > 0.01);
        train.finishTraining();

        return new SvmPredictor(svm);
    }
}
