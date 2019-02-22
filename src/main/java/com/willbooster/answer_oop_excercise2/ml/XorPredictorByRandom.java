package com.willbooster.answer_oop_excercise2.ml;

import org.encog.ml.data.MLData;

import java.util.Random;

public class XorPredictorByRandom implements XorPredictor {
    private final Random random;

    public XorPredictorByRandom() {
        random = new Random();
    }

    public int predict(MLData input) {
        return random.nextInt(2);
    }
}
