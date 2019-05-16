package com.expleague.ml;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.*;
import com.expleague.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.expleague.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.ArraySeq;
import com.expleague.commons.seq.IntSeqBuilder;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.cli.builders.data.impl.DataBuilderCrossValidation;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.FeaturesTxtPool;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.Linear;
import com.expleague.ml.func.NormalizedLinear;
import com.expleague.ml.loss.*;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.meta.items.QURLItem;
import com.expleague.ml.methods.*;
import com.expleague.ml.methods.greedyRegion.*;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.ModelTools;
import com.expleague.ml.models.ObliviousTree;
import com.expleague.ml.models.pgm.ProbabilisticGraphicalModel;
import com.expleague.ml.models.pgm.Route;
import com.expleague.ml.models.pgm.SimplePGM;
import gnu.trove.map.hash.TDoubleDoubleHashMap;
import gnu.trove.map.hash.TDoubleIntHashMap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.IntStream;

import static com.expleague.commons.math.MathTools.sqr;
import static com.expleague.commons.math.vectors.VecTools.copy;
import static com.expleague.ml.cli.builders.data.ReaderFactory.createFeatureTxtReader;
import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 26.11.12
 *
 * Time: 15:50
 */
public class BinarizeTests extends GridTest {
    private FastRandom rng;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        rng = new FastRandom(0);
    }

    public class addBoostingListeners<GlobalLoss extends TargetFunc> {
        addBoostingListeners(final GradientBoosting<GlobalLoss> boosting, final GlobalLoss loss, final Pool<?> _learn, final Pool<?> _validate) {
            final Consumer counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void accept(final Trans partial) {
                    System.out.print("\n" + index++);
                }
            };
            final ScoreCalcer learnListener = new ScoreCalcer(/*"\tlearn:\t"*/"\t", _learn.vecData(), _learn.target(L2.class));
            final ScoreCalcer validateListener = new ScoreCalcer(/*"\ttest:\t"*/"\t", _validate.vecData(), _validate.target(L2.class));
            final Consumer<Trans> modelPrinter = new ModelPrinter();
            final Consumer<Trans> qualityCalcer = new QualityCalcer();
            boosting.addListener(counter);
            boosting.addListener(learnListener);
            boosting.addListener(validateListener);
            boosting.addListener(qualityCalcer);
            //boosting.addListener(modelPrinter);
            final Ensemble ans = boosting.fit(_learn.vecData(), loss);
      /*System.out.println();
      System.out.println(ans.models.length);
      for(int t = ans.models.length - 10; t < ans.models.length; t++) {
        System.out.println(ans.models[t].toString());
      }*/
            Vec current = new ArrayVec(_validate.size());
            for (int i = 0; i < _validate.size(); i++) {
                double f = 0;
                for (int j = 0; j < ans.models.length; j++)
                    f += ans.weights.get(j) * ((Func) ans.models[j]).value(_validate.vecData().data().row(i));
                current.set(i, f);
            }
            System.out.println("\n + Final loss = " + VecTools.distance(current, _validate.target(L2.class).target) / Math.sqrt(_validate.size()));

        }
    }

    /*public void testTreeOutput() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(learn.vecData(), 32), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
    }*/

    /*public void testOTBoost3() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(learn.vecData(), 32), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
    }*/

    public void testOTBoost4() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(learn.vecData(), 32, true), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
        //GridUtils.outArr();
    }

    public void testOTBoost5() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(learn.vecData(), 32, true), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
        //GridUtils.outArr();
    }

    public void testOTBoost6() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(learn.vecData(), 32, true), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
        //GridUtils.outArr();
    }

    public void testOTBoost7() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(learn.vecData(), 32, true), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
        //GridUtils.outArr();
    }

    public void testOTBoost8() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(learn.vecData(), 32, true), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
        //GridUtils.outArr();
    }

    public void testOTBoost9() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(learn.vecData(), 32, true), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
        //GridUtils.outArr();
    }

    public void testOTBoost10() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(learn.vecData(), 32, true), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
        //GridUtils.outArr();
    }

    public void testOTBoost11() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(learn.vecData(), 32, true), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
        //GridUtils.outArr();
    }

    /*
    public void testOTBoostRandomSplit16() {
        List<? extends Pool<?>> split = DataTools.splitDataSet(all, rng, 0.8, 0.2);
        Pool<?> lrn = split.get(0);
        Pool<?> vld = split.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(lrn.vecData(), 16, true), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, lrn.target(SatL2.class), lrn, vld);

        final GradientBoosting<SatL2> boosting2 = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(lrn.vecData(), 16), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting2, lrn.target(SatL2.class), lrn, vld);
        //GridUtils.outArr();
    }*/

    protected static class ScoreCalcer implements ProgressHandler {
        final String message;
        final Vec current;
        private final VecDataSet ds;
        private final TargetFunc target;

        public ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target) {
            this.message = message;
            this.ds = ds;
            this.target = target;
            current = new ArrayVec(ds.length());
        }

        double min = 1e10;

        @Override
        public void accept(final Trans partial) {
            if (partial instanceof Ensemble) {
                final Ensemble linear = (Ensemble) partial;
                final Trans increment = linear.last();
                for (int i = 0; i < ds.length(); i++) {
                    if (increment instanceof Ensemble) {
                        current.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
                    } else {
                        current.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
                    }
                }
            } else {
                for (int i = 0; i < ds.length(); i++) {
                    current.set(i, ((Func) partial).value(ds.data().row(i)));
                }
            }
            final double value = target.value(current);
            System.out.print(message + value);
            min = Math.min(value, min);
            System.out.print(" best = " + min);
        }
    }

    private static class ModelPrinter implements ProgressHandler {
        @Override
        public void accept(final Trans partial) {
            if (partial instanceof Ensemble) {
                final Ensemble model = (Ensemble) partial;
                final Trans increment = model.last();
                System.out.print("\t" + increment);
            }
        }
    }

    private class QualityCalcer implements ProgressHandler {
        Vec residues = VecTools.copy(learn.target(L2.class).target);
        double total = 0;
        int index = 0;

        @Override
        public void accept(final Trans partial) {
            if (partial instanceof Ensemble) {
                final Ensemble model = (Ensemble) partial;
                final Trans increment = model.last();

                final TDoubleIntHashMap values = new TDoubleIntHashMap();
                final TDoubleDoubleHashMap dispersionDiff = new TDoubleDoubleHashMap();
                int index = 0;
                final VecDataSet ds = learn.vecData();
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
                System.out.print("\tscore:\t" + score + "\tmean:\t" + (total / this.index));
            }
        }
    }
}


