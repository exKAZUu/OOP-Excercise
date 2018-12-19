package com.willbooster.oop_excercise3;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.svm.SVM;
import org.encog.ml.svm.training.SVMSearchTrain;

public class SvmPredictor {
    /**
     * The input necessary for XOR.
     */
    public static double XOR_INPUT[][] = {{0.0, 0.0}, {1.0, 0.0},
            {0.0, 1.0}, {1.0, 1.0}};

    /**
     * The ideal data necessary for XOR.
     */
    public static double XOR_IDEAL[][] = {{0.0}, {1.0}, {1.0}, {0.0}};

    private SVM svm;

    public SvmPredictor() {
    }

    public SVMSearchTrain train(MLDataSet trainingSet) {
        this.svm = new SVM(2, false);

        // train the neural network
        var train = new SVMSearchTrain(this.svm, trainingSet);

        var epoch = 1;
        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            epoch++;
        } while (train.getError() > 0.01);
        train.finishTraining();

        return train;
    }

    public MLData predict(MLData input) {
        return this.svm.compute(input);
    }

    public static void main(String[] args) {
        var trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

        var predictor = new SvmPredictor();
        predictor.train(trainingSet);
        System.out.println("SVM:");
        for (var pair : trainingSet) {
            var output = predictor.predict(pair.getInput());
            System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1)
                    + ", actual=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0));
        }
    }
}
