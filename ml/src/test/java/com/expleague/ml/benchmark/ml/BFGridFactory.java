package com.expleague.ml.benchmark.ml;

import com.expleague.ml.BFGrid;
import com.expleague.ml.GridTools;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.binarization.algorithms.*;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.Pool;
import javafx.application.Platform;
import javafx.scene.control.Label;

import static com.expleague.ml.binarization.algorithms.NaiveProbabilityGrid.probabilityGrid;
import static com.expleague.ml.binarization.algorithms.ProbabilityGridBigInt.probabilityGrid_bigInt;
import static com.expleague.ml.binarization.algorithms.ProbabilityGridMedianWarming.probabilityGrid_mixed;
import static com.expleague.ml.binarization.algorithms.ProbabilityGridPresort.probabilityGrid_presort;
import static com.expleague.ml.binarization.algorithms.ProbabilityGridWithMedianBest.probabilityGridMedian;

public class BFGridFactory {
    public static BFGrid makeGrid(MethodType type, Pool<?> pool, int binFactor, BuildProgressHandler buildProgressHandler, Label binTime) {
        VecDataSet data = pool.vecData();
        long s = System.nanoTime();
        BFGrid res;
        switch (type) {
            case MEDIAN:
                res =  GridTools.medianGrid(data, binFactor, buildProgressHandler);
                break;
            case EQUAL_FREQUENCY:
                res =  EqualFrequencyBinarization.equalFreqGrid(data, binFactor, buildProgressHandler);
                break;
            case EQUAL_WIDTH:
                res =  EqualWidthBinarization.equalWidthGrid(data, binFactor, buildProgressHandler);
                break;
            case MDLP:
                res = new EntropyDiscretization(pool, binFactor).fit();
                break;
            case D2:
                res = new D2Algorithm(pool, binFactor).fit();
                break;
            case PROBABILITY_FAST:
                res = probabilityGrid(data, binFactor, true, buildProgressHandler);
                break;
            case PROBABILITY_SIMPLE:
                res = probabilityGrid(data, binFactor, false, buildProgressHandler);
                break;
            case PROBABILITY_BIG_INT:
                res = probabilityGrid_bigInt(data, binFactor, buildProgressHandler);
                break;
            case PROBABILITY_MIXED:
                res = probabilityGrid_mixed(data, binFactor, true, buildProgressHandler);
                break;
            case PROBABILITY_PRESORT:
                res = probabilityGrid_presort(data, binFactor, buildProgressHandler);
                break;
            case PROBABILITY_MEDIAN:
                res = probabilityGridMedian(data, binFactor, buildProgressHandler);
                break;
            case PROBABILITY_PARALLEL_2:
                res = ProbabilityGridParallel.probabilityGrid(data, binFactor, buildProgressHandler, 2);
                break;
            case PROBABILITY_PARALLEL_4:
                res = ProbabilityGridParallel.probabilityGrid(data, binFactor, buildProgressHandler, 4);
                break;
            case PROBABILITY_PARALLEL_8:
                res = ProbabilityGridParallel.probabilityGrid(data, binFactor, buildProgressHandler, 8);
                break;
            default:
                res =  GridTools.medianGrid(data, binFactor, buildProgressHandler);
                break;

        }

        long f = System.nanoTime();
        long d = (f - s);

        Platform.runLater(() -> binTime.setText((d < 1000000000) ? ( d/ 1000000) + " ms." : ( d/ 1000000000) + " sec."));
        return res;
    }

    public static int getStepsCount(MethodType type, VecDataSet data, int binFactor) {
        switch (type) {
            case MEDIAN:
                return binFactor;
            case EQUAL_FREQUENCY:
                return binFactor;
            case EQUAL_WIDTH:
                return binFactor;
            case MDLP:
                return data.xdim();
            case D2:
                return data.xdim();
            case PROBABILITY_FAST:
                return binFactor * data.xdim() * data.xdim();
            case PROBABILITY_SIMPLE:
                return binFactor * data.xdim() * data.xdim();
            case PROBABILITY_BIG_INT:
                return binFactor * data.xdim() * data.xdim();
            case PROBABILITY_MIXED:
                return binFactor * data.xdim() * data.xdim();
            case PROBABILITY_PRESORT:
                return binFactor * data.xdim() * data.xdim();
            case PROBABILITY_MEDIAN:
                return binFactor * data.xdim() * data.xdim();
            case PROBABILITY_PARALLEL_2:
                return binFactor * data.xdim() * data.xdim();
            case PROBABILITY_PARALLEL_4:
                return binFactor * data.xdim() * data.xdim();
            case PROBABILITY_PARALLEL_8:
                return binFactor * data.xdim() * data.xdim();
        }

        return binFactor;
    }

    public static String getAlgorithmName(MethodType type) {
        switch (type) {
            case MEDIAN:
                return "Median division";
            case EQUAL_FREQUENCY:
                return "Equal freq";
            case EQUAL_WIDTH:
                return "Equal width";
            case MDLP:
                return "MDLP";
            case D2:
                return "D2";
            case PROBABILITY_FAST:
                return "Probability fast";
            case PROBABILITY_SIMPLE:
                return "Probability";
            case PROBABILITY_BIG_INT:
                return "Probability + BigInteger";
            case PROBABILITY_MIXED:
                return "Probability + median warming";
            case PROBABILITY_PRESORT:
                return "Probability + presorting features";
            case PROBABILITY_MEDIAN:
                return "Probability with median best";
            case PROBABILITY_PARALLEL_2:
                return "Probability parallel @2";
            case PROBABILITY_PARALLEL_4:
                return "Probability parallel @4";
            case PROBABILITY_PARALLEL_8:
                return "Probability parallel @8";
        }

        return "Median division";
    }

    public static String[] getAlgorithmNames() {
        String[] res = new String[MethodType.values().length];
        int ptr = 0;
        for (MethodType type : MethodType.values()) {
            res[ptr++] = getAlgorithmName(type);
        }

        return res;
    }

    public static MethodType getAlgorithmType(String raw) {
        switch (raw) {
            case "Median division":
                return MethodType.MEDIAN;
            case "Equal freq":
                return MethodType.EQUAL_FREQUENCY;
            case "Equal width":
                return MethodType.EQUAL_WIDTH;
            case "MDLP":
                return MethodType.MDLP;
            case "D2":
                return MethodType.D2;
            case "Probability fast":
                return MethodType.PROBABILITY_FAST;
            case "Probability":
                return MethodType.PROBABILITY_SIMPLE;
            case "Probability + BigInteger":
                return MethodType.PROBABILITY_BIG_INT;
            case "Probability + median warming":
                return MethodType.PROBABILITY_MIXED;
            case "Probability + presorting features":
                return MethodType.PROBABILITY_PRESORT;
            case "Probability with median best":
                return MethodType.PROBABILITY_MEDIAN;
            case "Probability parallel @2":
                return MethodType.PROBABILITY_PARALLEL_2;
            case "Probability parallel @4":
                return MethodType.PROBABILITY_PARALLEL_4;
            case "Probability parallel @8":
                return MethodType.PROBABILITY_PARALLEL_8;
            default:
                return MethodType.MEDIAN;
        }
    }
}
