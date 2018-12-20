package com.willbooster.oop_excercise1.ml;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;

public interface XorOperator {
    void train(MLDataSet trainingSet);

    int predict(MLData input);
}
