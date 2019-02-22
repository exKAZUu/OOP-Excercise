package com.willbooster.oop_excercise6;

import org.encog.ml.data.MLData;

public abstract class XorPredictor {
    public abstract int predict(MLData input);
    protected void validateInput(MLData input) {
        if (input.size() != 2) {
            throw new IllegalArgumentException("入力データの個数が2個ではありません。");
        }
        if ((input.getData(0) != 0.0 && input.getData(0) != 1.0) ||
                (input.getData(1) != 0.0 && input.getData(1) != 1.0)) {
            throw new IllegalArgumentException("入力データが0または1ではありません。");
        }
    }
}
