package com.expleague.ml.binarization.algorithms;

import com.expleague.ml.data.tools.Pool;

import java.lang.reflect.Array;
import java.util.Arrays;

public class EntropyDiscretization {
    private Pool<?> learn;
    private int binFactor;
    private int[] classes;

    public EntropyDiscretization(Pool<?> learn, int binFactor) {
        this.learn = learn;
        this.binFactor = binFactor;

        int targetsLength = learn.target(0).length();
        double[] targets = new double[targetsLength];
        double[] sortedTargets = new double[targetsLength];
        for (int i = 0; i < targetsLength; i++) {
            targets[i] = Double.valueOf(learn.target(0).at(i).toString());
            sortedTargets[i] = targets[i];
        }

        Arrays.sort(sortedTargets);

        double min = sortedTargets[0];
        double max = sortedTargets[targetsLength - 1];

        double width = (max - min) / binFactor;

        int[] classes = new int[targetsLength];

        for (int i = 0; i < targetsLength; i++) {
            classes[i] = binNumber(min, targets[i], width);
        }

        this.classes = classes;
    }

    private static int binNumber(double min, double val, double width) {
        return  (int)Math.round((val - min) / width);
    }
}
