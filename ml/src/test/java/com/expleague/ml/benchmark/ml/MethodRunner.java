package com.expleague.ml.benchmark.ml;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.GridTools;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import javafx.scene.chart.XYChart;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.List;

public class MethodRunner {
    private String logFileName;
    private Pool<?> dataset;
    private FastRandom rng;
    private MethodType type;
    private double learnSize;
    private XYChart.Series convergeSeries;
    private int binFactor;
    private int treeDepth;
    private int iterationsCount;
    private double step;

    public MethodRunner(String logFileName,
                        Pool<?> dataset,
                        FastRandom rng,
                        MethodType type,
                        double learnSize,
                        XYChart.Series convergeSeries,
                        int binFactor,
                        int treeDepth,
                        int iterationsCount,
                        double step) {

        this.logFileName = logFileName;
        this.dataset = dataset;
        this.rng = rng;
        this.type = type;
        this.learnSize = learnSize;
        this.convergeSeries = convergeSeries;
        this.binFactor = binFactor;
        this.treeDepth = treeDepth;
        this.iterationsCount = iterationsCount;
        this.step = step;
    }

    public void run() throws Exception {
        PrintWriter logger = new PrintWriter(new FileWriter(logFileName));

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(dataset, rng, learnSize, 1 - learnSize);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(BFGridFactory.makeGrid(type, local_learn.vecData(), binFactor), treeDepth), rng),
                L2Reg.class, iterationsCount, step
        );

        new AddBoostingListeners<>(boosting, local_learn.target(SatL2.class), dataset, local_learn, local_validate, logger, convergeSeries);
        logger.close();
    }
}
