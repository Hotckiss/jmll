package com.expleague.ml.benchmark.calcers;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.L2;
import gnu.trove.map.hash.TDoubleDoubleHashMap;
import gnu.trove.map.hash.TDoubleIntHashMap;

import java.io.PrintWriter;

import static com.expleague.commons.math.MathTools.sqr;
import static java.lang.Math.log;

public class BenchmarkQualityCalcer implements ProgressHandler {
    Pool<?> dataset;
    Vec residues;
    double total = 0;
    int index = 0;
    private final PrintWriter printWriter;

    public BenchmarkQualityCalcer(final PrintWriter printWriter, Pool<?> dataset) {
        this.printWriter = printWriter;
        this.dataset = dataset;
        residues = VecTools.copy(dataset.target(L2.class).target);
    }

    @Override
    public void accept(final Trans partial) {
        if (partial instanceof Ensemble) {
            final Ensemble model = (Ensemble) partial;
            final Trans increment = model.last();

            final TDoubleIntHashMap values = new TDoubleIntHashMap();
            final TDoubleDoubleHashMap dispersionDiff = new TDoubleDoubleHashMap();
            int index = 0;
            final VecDataSet ds = dataset.vecData();
            for (int i = 0; i < ds.data().rows(); i++) {
                final double value;
                if (increment instanceof Ensemble) {
                    value = increment.trans(ds.data().row(i)).get(0);
                } else {
                    value = ((Func) increment).value(ds.data().row(i));
                }
                values.adjustOrPutValue(value, 1, 1);
                final double ddiff = sqr(residues.get(index)) - sqr(residues.get(index) - value);
                residues.adjust(index, -model.wlast() * value);
                dispersionDiff.adjustOrPutValue(value, ddiff, ddiff);
                index++;
            }
//          double totalDispersion = VecTools.multiply(residues, residues);
            double score = 0;
            for (final double key : values.keys()) {
                final double regularizer = 1 - 2 * log(2) / log(values.get(key) + 1);
                score += dispersionDiff.get(key) * regularizer;
            }
//          score /= totalDispersion;
            total += score;
            this.index++;
            printWriter.print("\tscore:\t" + score + "\tmean:\t" + (total / this.index));
            System.out.print("\tscore:\t" + score + "\tmean:\t" + (total / this.index));
        }
    }
}
