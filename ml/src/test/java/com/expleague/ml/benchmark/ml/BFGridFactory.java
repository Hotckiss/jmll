package com.expleague.ml.benchmark.ml;

import com.expleague.ml.BFGrid;
import com.expleague.ml.GridTools;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.data.set.VecDataSet;
import javafx.application.Platform;
import javafx.scene.control.Label;

public class BFGridFactory {
    public static BFGrid makeGrid(MethodType type, VecDataSet data, int binFactor, BuildProgressHandler buildProgressHandler, Label binTime) {
        long s = System.nanoTime();
        BFGrid res;
        switch (type) {
            case MEDIAN:
                res =  GridTools.medianGrid(data, binFactor, buildProgressHandler);
                break;
            case PROBABILITY_FAST:
                res = GridTools.probabilityGrid(data, binFactor, true, buildProgressHandler);
                break;
            case PROBABILITY_SIMPLE:
                res = GridTools.probabilityGrid(data, binFactor, false, buildProgressHandler);
                break;
            case PROBABILITY_BIG_INT:
                res = GridTools.probabilityGrid_bigInt(data, binFactor, buildProgressHandler);
                break;
            case PROBABILITY_MIXED:
                res = GridTools.probabilityGrid_mixed(data, binFactor, true, buildProgressHandler);
                break;
            case PROBABILITY_PRESORT:
                res = GridTools.probabilityGrid_presort(data, binFactor, buildProgressHandler);
                break;
            case PROBABILITY_MEDIAN:
                res = GridTools.probabilityGridMedian(data, binFactor, buildProgressHandler);
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
        }

        return binFactor;
    }

    public static String getAlgorithmName(MethodType type) {
        switch (type) {
            case MEDIAN:
                return "Median division";
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
            default:
                return MethodType.MEDIAN;
        }
    }
}
