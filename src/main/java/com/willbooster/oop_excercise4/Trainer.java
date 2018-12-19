package com.willbooster.oop_excercise4;

import org.encog.ml.data.MLDataSet;

public interface Trainer {
    Predictor train(MLDataSet trainingSet);
}
