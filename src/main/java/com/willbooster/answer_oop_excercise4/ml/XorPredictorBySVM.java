package com.willbooster.answer_oop_excercise4.ml;

import com.willbooster.answer_oop_excercise4.XorPredictor;
import org.encog.ml.data.MLData;
import org.encog.ml.svm.SVM;

public class XorPredictorBySVM implements XorPredictor {
    private SVM svm;

    public XorPredictorBySVM(SVM svm) {
        this.svm = svm;
    }

    public int predict(MLData input) {
        MLData data = this.svm.compute(input);
        return data.getData(0) <= 0.5 ? 0 : 1;
    }
}
