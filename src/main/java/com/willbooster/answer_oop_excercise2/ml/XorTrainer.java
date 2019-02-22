package com.willbooster.answer_oop_excercise2.ml;

import org.encog.ml.data.MLDataSet;

public interface XorTrainer {
    XorPredictor train(MLDataSet trainingSet);
}
