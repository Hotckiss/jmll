package com.expleague.ml.benchmark.ml;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.benchmark.calcers.BenchmarkLearnScoreCalcer;
import com.expleague.ml.benchmark.calcers.BenchmarkModelPrinter;
import com.expleague.ml.benchmark.calcers.BenchmarkQualityCalcer;
import com.expleague.ml.benchmark.calcers.BenchmarkValidateScoreCalcer;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.GradientBoosting;
import javafx.scene.chart.XYChart;

import java.io.PrintWriter;
import java.util.function.Consumer;

public class AddBoostingListeners<GlobalLoss extends TargetFunc> {
    private BenchmarkLearnScoreCalcer benchmarkLearnScoreCalcer;
    private BenchmarkValidateScoreCalcer benchmarkValidateScoreCalcer;
    private Consumer<Trans> modelPrinter;
    private Consumer<Trans> qualityCalcer;
    private Consumer counter;
    public AddBoostingListeners(final GradientBoosting<GlobalLoss> boosting,
                         final GlobalLoss loss,
                         final Pool<?> dataset,
                         final Pool<?> _learn,
                         final Pool<?> _validate,
                         final PrintWriter printWriter,
                         XYChart.Series series) {
        final Consumer counter = new ProgressHandler() {
            int index = 0;

            @Override
            public void accept(final Trans partial) {
                System.out.print("\n" + index);
                printWriter.print("\n" + index++);
            }
        };
        final BenchmarkLearnScoreCalcer learnListener = new BenchmarkLearnScoreCalcer(/*"\tlearn:\t"*/"\t", _learn.vecData(), _learn.target(L2.class), printWriter);
        final BenchmarkValidateScoreCalcer validateListener = new BenchmarkValidateScoreCalcer(/*"\ttest:\t"*/"\t", _validate.vecData(), _validate.target(L2.class), printWriter, series);
        final Consumer<Trans> modelPrinter = new BenchmarkModelPrinter();
        final Consumer<Trans> qualityCalcer = new BenchmarkQualityCalcer(printWriter, dataset);
        this.counter = counter;
        this.benchmarkLearnScoreCalcer = learnListener;
        this.benchmarkValidateScoreCalcer = validateListener;
        this.modelPrinter = modelPrinter;
        this.qualityCalcer = qualityCalcer;
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