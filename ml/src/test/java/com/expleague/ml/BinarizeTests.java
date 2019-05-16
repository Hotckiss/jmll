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

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
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
        addBoostingListeners(final GradientBoosting<GlobalLoss> boosting, final GlobalLoss loss, final Pool<?> _learn, final Pool<?> _validate, final PrintWriter printWriter) {
            final Consumer counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void accept(final Trans partial) {
                    System.out.print("\n" + index);
                    printWriter.print("\n" + index++);
                }
            };
            final ScoreCalcer learnListener = new ScoreCalcer(/*"\tlearn:\t"*/"\t", _learn.vecData(), _learn.target(L2.class), printWriter);
            final ScoreCalcer validateListener = new ScoreCalcer(/*"\ttest:\t"*/"\t", _validate.vecData(), _validate.target(L2.class), printWriter);
            final Consumer<Trans> modelPrinter = new ModelPrinter();
            final Consumer<Trans> qualityCalcer = new QualityCalcer(printWriter);
            boosting.addListener(counter);
            boosting.addListener(learnListener);
            boosting.addListener(validateListener);
            boosting.addListener(qualityCalcer);
            //boosting.addListener(modelPrinter);
            final Ensemble ans = boosting.fit(_learn.vecData(), loss);
            Vec current = new ArrayVec(_validate.size());
            for (int i = 0; i < _validate.size(); i++) {
                double f = 0;
                for (int j = 0; j < ans.models.length; j++)
                    f += ans.weights.get(j) * ((Func) ans.models[j]).value(_validate.vecData().data().row(i));
                current.set(i, f);
            }
            System.out.println("\n + Final loss = " + VecTools.distance(current, _validate.target(L2.class).target) / Math.sqrt(_validate.size()) + "\n");
            printWriter.println("\n + Final loss = " + VecTools.distance(current, _validate.target(L2.class).target) / Math.sqrt(_validate.size()) + "\n");
        }
    }

    public void testOTBoost1Prob() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread1LogProb.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(1);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(local_learn.vecData(), 32, true), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost2Prob() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread2LogProb.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(2);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(local_learn.vecData(), 32, true), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost3Prob() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread3LogProb.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(3);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(local_learn.vecData(), 32, true), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost4Prob() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread4LogProb.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(4);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(local_learn.vecData(), 32, true), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost5Prob() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread5LogProb.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(5);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(local_learn.vecData(), 32, true), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost6Prob() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread6LogProb.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(6);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(local_learn.vecData(), 32, true), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost7Prob() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread7LogProb.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(7);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(local_learn.vecData(), 32, true), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost8Prob() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread8LogProb.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(8);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid(local_learn.vecData(), 32, true), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost1Med() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread1LogMed.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(1);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(local_learn.vecData(), 32), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost2Med() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread2LogMed.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(2);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(local_learn.vecData(), 32), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost3Med() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread3LogMed.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(3);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(local_learn.vecData(), 32), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost4Med() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread4LogMed.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(4);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(local_learn.vecData(), 32), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost5Med() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread5LogMed.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(5);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(local_learn.vecData(), 32), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost6Med() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread6LogMed.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(6);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(local_learn.vecData(), 32), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost7Med() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread7LogMed.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(7);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(local_learn.vecData(), 32), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost8Med() {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread8LogMed.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        FastRandom rand = new FastRandom(8);
        List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rand, 0.1, 0.9);
        Pool<?> local_all = split.get(0);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(local_all, rand, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(local_learn.vecData(), 32), 6), rand),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter);
        printWriter.println();

        printWriter.close();
    }

    protected static class ScoreCalcer implements ProgressHandler {
        final String message;
        final Vec current;
        private final VecDataSet ds;
        private final TargetFunc target;
        private final PrintWriter printWriter;

        public ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target, final PrintWriter printWriter) {
            this.message = message;
            this.ds = ds;
            this.target = target;
            this.printWriter = printWriter;
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
            printWriter.print(message + value);
            min = Math.min(value, min);
            System.out.print(" best = " + min);
            printWriter.print(" best = " + min);
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
        private final PrintWriter printWriter;

        public QualityCalcer(final PrintWriter printWriter) {
            this.printWriter = printWriter;
        }

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
                printWriter.print("\tscore:\t" + score + "\tmean:\t" + (total / this.index));
                System.out.print("\tscore:\t" + score + "\tmean:\t" + (total / this.index));
            }
        }
    }
}


