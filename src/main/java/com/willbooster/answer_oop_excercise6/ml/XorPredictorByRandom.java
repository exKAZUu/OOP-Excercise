package com.willbooster.answer_oop_excercise6.ml;

import com.willbooster.answer_oop_excercise6.XorPredictor;
import org.encog.ml.data.MLData;

import java.util.Random;

public class XorPredictorByRandom extends XorPredictor {
    private final Random random;

    public XorPredictorByRandom() {
        random = new Random();
    }

    public int predictBody(MLData input) {
        return random.nextInt(2);
    }
}
