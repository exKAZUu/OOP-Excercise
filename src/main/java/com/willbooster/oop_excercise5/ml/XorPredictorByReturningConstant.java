package com.willbooster.oop_excercise5.ml;

import com.willbooster.oop_excercise5.XorPredictor;
import org.encog.ml.data.MLData;

public class XorPredictorByReturningConstant implements XorPredictor {
    public int predict(MLData input) {
        return 0;
    }
}
