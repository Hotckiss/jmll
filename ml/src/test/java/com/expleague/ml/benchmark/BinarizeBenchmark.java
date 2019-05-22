package com.expleague.ml.benchmark;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.GridTools;
import com.expleague.ml.benchmark.ml.AddBoostingListeners;
import com.expleague.ml.benchmark.ml.MethodRunner;
import com.expleague.ml.benchmark.ml.MethodType;
import com.expleague.ml.benchmark.ui.BinarizeBenchmarkUIUtils;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.testUtils.TestResourceLoader;
import javafx.application.Application;
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

import static com.expleague.commons.math.MathTools.sqr;

public class BinarizeBenchmark extends Application {
    public static Pool<?> dataset;
    private static FastRandom rng = new FastRandom(0);
    XYChart.Series series1 = new XYChart.Series();

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


        GridPane gridpane = BinarizeBenchmarkUIUtils.makeGridPane();
        BinarizeBenchmarkUIUtils.addInputs(gridpane);

        Button runButt = new Button("Run!");
        GridPane.setHalignment(runButt, HPos.RIGHT);
        gridpane.add(runButt, 1, 3);

        runButt.setOnAction(event -> {
            new Thread(() -> {
                loadDataSet();
                try {
                    new MethodRunner("Algorithm1Log.txt",
                            dataset,
                            rng,
                            MethodType.MEDIAN,
                            0.2,
                            series1,
                            32,
                            6,
                            2000,
                            0.005).run();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
        });

        series1.setName("Algorithm 1");
        BinarizeBenchmarkUIUtils.addAlgorithm1Chart(gridpane, series1);

        root.getChildren().add(gridpane);
        primaryStage.setScene(new Scene(root, 800, 600));
        primaryStage.show();
    }
}
