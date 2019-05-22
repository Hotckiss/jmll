package com.expleague.ml;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.benchmark.calcers.BenchmarkLearnScoreCalcer;
import com.expleague.ml.benchmark.calcers.BenchmarkModelPrinter;
import com.expleague.ml.benchmark.calcers.BenchmarkQualityCalcer;
import com.expleague.ml.benchmark.calcers.BenchmarkValidateScoreCalcer;
import com.expleague.ml.benchmark.ui.BinarizeBenchmarkUIUtils;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.testUtils.TestResourceLoader;
import gnu.trove.map.hash.TDoubleDoubleHashMap;
import gnu.trove.map.hash.TDoubleIntHashMap;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.HPos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.*;
import javafx.stage.Stage;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static com.expleague.commons.math.MathTools.sqr;
import static java.lang.Math.log;

public class BinarizeBenchmark extends Application {
    public static Pool<?> dataset;
    private static FastRandom rng = new FastRandom(0);
    ArrayList<Thread> t = new ArrayList<>();
    private static synchronized void loadDataSet() {
        try {
            dataset = TestResourceLoader.loadPool("features.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Binarization Benchmark");

        StackPane root = new StackPane();
        XYChart.Series series = new XYChart.Series();

        GridPane gridpane = BinarizeBenchmarkUIUtils.makeGridPane();
        addInputs(gridpane);

        Button runButt = new Button("Run!");
        // run button
        GridPane.setHalignment(runButt, HPos.RIGHT);
        gridpane.add(runButt, 1, 3);

        runButt.setOnAction(event -> {
            Thread th = new Thread(() -> {
                loadDataSet();
                testOTBoost8Prob(series);
            });
            t.add(th);
            th.start();
        });

        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Iteration");
        final LineChart<Number,Number> lineChart =
                new LineChart<Number,Number>(xAxis,yAxis);

        lineChart.setTitle("First binarization");

        series.setName("Med");

        lineChart.setCreateSymbols(false);
        lineChart.setStyle(".chart-series-line { -fx-stroke-width: 1px; }");
        lineChart.setStyle(".default-color0.chart-series-line { -fx-stroke: #e9967a; }");
        lineChart.getData().add(series);
        gridpane.add(lineChart, 0, 4, 2, 1);


        root.getChildren().add(gridpane);
        primaryStage.setScene(new Scene(root, 800, 600));
        primaryStage.show();
    }

    private void addInputs(GridPane gridpane) {
        Label datasetLabel = new Label("Datset path:");
        TextField datasetInput = new TextField();
        BinarizeBenchmarkUIUtils.addDatasetInput(gridpane, datasetLabel, datasetInput);

        Label featuresLabel = new Label("Features to extract:");
        TextField featuresInput = new TextField();
        BinarizeBenchmarkUIUtils.addFeaturesInput(gridpane, featuresLabel, featuresInput);

        Label targetColumnLabel = new Label("Target column:");
        TextField targetInput = new TextField();
        BinarizeBenchmarkUIUtils.addTargetInput(gridpane, targetColumnLabel, targetInput);
    }

    public class addBoostingListeners<GlobalLoss extends TargetFunc> {
        addBoostingListeners(final GradientBoosting<GlobalLoss> boosting, final GlobalLoss loss, final Pool<?> _learn, final Pool<?> _validate, final PrintWriter printWriter, XYChart.Series series) {
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

    public void testOTBoost8Prob(XYChart.Series series) {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread8LogProb.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(dataset, rng, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(local_learn.vecData(), 8), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter, series);
        printWriter.println();

        printWriter.close();
    }

    public void testOTBoost8Med(XYChart.Series series) {
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("thread8LogMed.txt");
        } catch (Exception ex) {
        }

        PrintWriter printWriter = new PrintWriter(fileWriter);

        List<? extends Pool<?>> split_local_all = DataTools.splitDataSet(dataset, rng, 0.2, 0.8);
        Pool<?> local_learn = split_local_all.get(0);
        Pool<?> local_validate = split_local_all.get(1);

        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
                new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.probabilityGrid_presort(local_learn.vecData(), 8), 6), rng),
                L2Reg.class, 2000, 0.005
        );
        new addBoostingListeners<>(boosting, local_learn.target(SatL2.class), local_learn, local_validate, printWriter, series);
        printWriter.println();

        printWriter.close();
    }
}
