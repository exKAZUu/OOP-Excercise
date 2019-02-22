package com.willbooster.answer_oop_excercise3.ml;

import org.encog.ml.data.MLDataSet;

public interface XorTrainer {
    XorPredictor train(MLDataSet trainingSet);
}
