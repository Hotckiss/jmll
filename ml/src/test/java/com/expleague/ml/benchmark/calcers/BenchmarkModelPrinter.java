package com.expleague.ml.benchmark.calcers;

import com.expleague.commons.math.Trans;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.func.Ensemble;

public class BenchmarkModelPrinter implements ProgressHandler {
    @Override
    public void accept(final Trans partial) {
        if (partial instanceof Ensemble) {
            final Ensemble model = (Ensemble) partial;
            final Trans increment = model.last();
            System.out.print("\t" + increment);
        }
    }
}