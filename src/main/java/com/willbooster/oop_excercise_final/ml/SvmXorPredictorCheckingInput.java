package com.willbooster.oop_excercise_final.ml;

import com.willbooster.oop_excercise_final.XorPredictor;
import org.encog.ml.data.MLData;
import org.encog.ml.svm.SVM;

public class SvmXorPredictorCheckingInput implements XorPredictor {
    private SVM svm;

    public SvmXorPredictorCheckingInput(SVM svm) {
        this.svm = svm;
    }

    public MLData predict(MLData input) {
        if (input.getData(0) != 0.0 && input.getData(0) != 1.0) {
            throw new RuntimeException("1");
        }
        if (input.getData(1) != 0.0 && input.getData(1) != 1.0) {
            throw new RuntimeException("2");
        }
        return this.svm.compute(input);
    }
}
