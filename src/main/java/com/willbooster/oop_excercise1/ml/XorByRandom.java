package com.willbooster.oop_excercise1.ml;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;

import java.util.Random;

public class XorByRandom implements XorOperator {
    private final Random random;

    public XorByRandom() {
        random = new Random();
    }

    public void train(MLDataSet trainingSet) {
    }

    public int predict(MLData input) {
        return random.nextInt(2);
    }
}
