package com.willbooster.oop_excercise_final.ml;

import com.willbooster.oop_excercise_final.XorPredictor;
import org.encog.ml.data.MLData;
import org.encog.ml.svm.SVM;

public class SvmXorPredictor implements XorPredictor {
    private SVM svm;

    public SvmXorPredictor(SVM svm) {
        this.svm = svm;
    }

    public MLData predict(MLData input) {
        return this.svm.compute(input);
    }
}
