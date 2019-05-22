package com.expleague.ml.benchmark.ml;

import com.expleague.ml.BFGrid;
import com.expleague.ml.GridTools;
import com.expleague.ml.data.set.VecDataSet;

public class BFGridFactory {
    public static BFGrid makeGrid(MethodType type, VecDataSet data, int binFactor) {
        switch (type) {
            case MEDIAN:
                return GridTools.medianGrid(data, binFactor);
            case PROBABILITY_FAST:
                return GridTools.probabilityGrid(data, binFactor, true);
            case PROBABILITY_SIMPLE:
                return GridTools.probabilityGrid(data, binFactor, false);
            case PROBABILITY_BIG_INT:
                return GridTools.probabilityGrid_bigInt(data, binFactor);
            case PROBABILITY_MIXED:
                return GridTools.probabilityGrid_mixed(data, binFactor, true);
            case PROBABILITY_PRESORT:
                return GridTools.probabilityGrid_presort(data, binFactor);
            case PROBABILITY_MEDIAN:
                return GridTools.probabilityGridMedian(data, binFactor);
        }

        return GridTools.medianGrid(data, binFactor);
    }

}
