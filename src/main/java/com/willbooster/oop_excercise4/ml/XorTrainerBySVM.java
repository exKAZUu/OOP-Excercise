package com.willbooster.oop_excercise4.ml;

import org.encog.ml.data.MLDataSet;
import org.encog.ml.svm.SVM;
import org.encog.ml.svm.training.SVMSearchTrain;

public class XorTrainerBySVM implements XorTrainer {
    public XorPredictorBySVM train(MLDataSet trainingSet) {
        // SVMの初期化
        var svm = new SVM(2, false);

        // 訓練データでSVMをトレーニング
        var train = new SVMSearchTrain(svm, trainingSet);
        var epoch = 1;
        do {
            train.iteration();
            epoch++;
        } while (train.getError() > 0.01 && epoch < 100);
        train.finishTraining();

        return new XorPredictorBySVM(svm);
    }
}
