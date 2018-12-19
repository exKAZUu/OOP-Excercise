package com.willbooster.oop_excercise4;

import org.encog.ml.data.MLData;

public interface Predictor {
    MLData predict(MLData input);
}
