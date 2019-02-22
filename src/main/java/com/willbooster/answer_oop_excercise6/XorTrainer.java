package com.willbooster.answer_oop_excercise6;

import org.encog.ml.data.MLDataSet;

public interface XorTrainer {
    XorPredictor train(MLDataSet trainingSet);
}
