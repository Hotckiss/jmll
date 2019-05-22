package com.expleague.ml.benchmark.ml;

import com.expleague.ml.BFGrid;
import com.expleague.ml.GridTools;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.data.set.VecDataSet;

public class BFGridFactory {
    public static BFGrid makeGrid(MethodType type, VecDataSet data, int binFactor, BuildProgressHandler buildProgressHandler) {
        switch (type) {
            case MEDIAN:
                return GridTools.medianGrid(data, binFactor, buildProgressHandler);
            case PROBABILITY_FAST:
                return GridTools.probabilityGrid(data, binFactor, true, buildProgressHandler);
            case PROBABILITY_SIMPLE:
                return GridTools.probabilityGrid(data, binFactor, false, buildProgressHandler);
            case PROBABILITY_BIG_INT:
                return GridTools.probabilityGrid_bigInt(data, binFactor, buildProgressHandler);
            case PROBABILITY_MIXED:
                return GridTools.probabilityGrid_mixed(data, binFactor, true, buildProgressHandler);
            case PROBABILITY_PRESORT:
                return GridTools.probabilityGrid_presort(data, binFactor, buildProgressHandler);
            case PROBABILITY_MEDIAN:
                return GridTools.probabilityGridMedian(data, binFactor, buildProgressHandler);
        }

        return GridTools.medianGrid(data, binFactor, buildProgressHandler);
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

}
