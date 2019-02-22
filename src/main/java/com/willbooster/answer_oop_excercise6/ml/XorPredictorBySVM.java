package com.willbooster.answer_oop_excercise6.ml;

import com.willbooster.answer_oop_excercise6.XorPredictor;
import org.encog.ml.data.MLData;
import org.encog.ml.svm.SVM;

public class XorPredictorBySVM extends XorPredictor {
    private SVM svm;

    public XorPredictorBySVM(SVM svm) {
        this.svm = svm;
    }

    public int predictBody(MLData input) {
        MLData data = this.svm.compute(input);
        return data.getData(0) <= 0.5 ? 0 : 1;
    }
}
