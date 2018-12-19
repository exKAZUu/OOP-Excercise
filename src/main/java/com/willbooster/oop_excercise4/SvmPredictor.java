package com.willbooster.oop_excercise4;

import org.encog.ml.data.MLData;
import org.encog.ml.svm.SVM;

public class SvmPredictor implements Predictor {
    private SVM svm;

    public SvmPredictor(SVM svm) {
        this.svm = svm;
    }

    public MLData predict(MLData input) {
        return this.svm.compute(input);
    }
}
