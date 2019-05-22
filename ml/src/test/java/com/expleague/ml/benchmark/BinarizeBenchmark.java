package com.expleague.ml.benchmark;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.benchmark.ml.BFGridFactory;
import com.expleague.ml.benchmark.ml.MethodRunner;
import com.expleague.ml.benchmark.ml.MethodType;
import com.expleague.ml.benchmark.ui.BinarizeBenchmarkUIUtils;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.testUtils.TestResourceLoader;
import javafx.application.Application;
import javafx.geometry.HPos;
import javafx.scene.Scene;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.ProgressBar;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

import java.io.IOException;

public class BinarizeBenchmark extends Application {
    public static Pool<?> dataset;
    private XYChart.Series series1 = new XYChart.Series();
    private XYChart.Series series2 = new XYChart.Series();
    private XYChart.Series series3 = new XYChart.Series();
    private XYChart.Series series4 = new XYChart.Series();
    private XYChart.Series series1T = new XYChart.Series();
    private XYChart.Series series2T = new XYChart.Series();
    private XYChart.Series series3T = new XYChart.Series();
    private XYChart.Series series4T = new XYChart.Series();

    private ProgressBar algorithm1Bar = new ProgressBar();
    private ProgressBar algorithm2Bar = new ProgressBar();
    private ProgressBar algorithm3Bar = new ProgressBar();
    private ProgressBar algorithm4Bar = new ProgressBar();

    private static synchronized void loadDataSet() {
        try {
            dataset = TestResourceLoader.loadPool("features.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        loadDataSet();
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Binarization Benchmark");

        StackPane root = new StackPane();

        GridPane gridpane = BinarizeBenchmarkUIUtils.makeGridPane();
        BinarizeBenchmarkUIUtils.addInputs(gridpane);

        algorithm1Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 0);
        algorithm2Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 1);
        algorithm3Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 2);
        algorithm4Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 3);

        Button runButt = new Button("Run!");
        GridPane.setHalignment(runButt, HPos.RIGHT);
        gridpane.add(runButt, 1, 3);

        runButt.setOnAction(event -> {
            new Thread(() -> {
                try {
                    new MethodRunner("Algorithm1Log.txt",
                            dataset,
                            new FastRandom(1),
                            MethodType.MEDIAN,
                            0.2,
                            series1,
                            series1T,
                            32,
                            6,
                            2000,
                            0.005,
                            new BuildProgressHandler(algorithm1Bar, BFGridFactory.getStepsCount(MethodType.MEDIAN, dataset.vecData(), 32))).run();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();

            new Thread(() -> {
                try {
                    new MethodRunner("Algorithm2Log.txt",
                            dataset,
                            new FastRandom(1),
                            MethodType.PROBABILITY_PRESORT,
                            0.2,
                            series2,
                            series2T,
                            32,
                            6,
                            2000,
                            0.005,
                            new BuildProgressHandler(algorithm2Bar, BFGridFactory.getStepsCount(MethodType.PROBABILITY_PRESORT, dataset.vecData(), 32))).run();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();

            new Thread(() -> {
                try {
                    new MethodRunner("Algorithm3Log.txt",
                            dataset,
                            new FastRandom(1),
                            MethodType.PROBABILITY_FAST,
                            0.2,
                            series3,
                            series3T,
                            32,
                            6,
                            2000,
                            0.005,
                            new BuildProgressHandler(algorithm3Bar, BFGridFactory.getStepsCount(MethodType.PROBABILITY_PRESORT, dataset.vecData(), 32))).run();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();

            new Thread(() -> {
                try {
                    new MethodRunner("Algorithm4Log.txt",
                            dataset,
                            new FastRandom(1),
                            MethodType.PROBABILITY_MEDIAN,
                            0.2,
                            series4,
                            series4T,
                            32,
                            6,
                            2000,
                            0.005,
                            new BuildProgressHandler(algorithm4Bar, BFGridFactory.getStepsCount(MethodType.PROBABILITY_PRESORT, dataset.vecData(), 32))).run();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
        });

        series1.setName("Algorithm 1");
        series2.setName("Algorithm 2");
        series3.setName("Algorithm 3");
        series4.setName("Algorithm 4");
        series1T.setName("Algorithm 1 train");
        series2T.setName("Algorithm 2 train");
        series3T.setName("Algorithm 3 train");
        series4T.setName("Algorithm 4 train");
        BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series1, series1T, 0);
        BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series2, series2T, 1);
        BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series3, series3T, 2);
        BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series4, series4T, 3);

        root.getChildren().add(gridpane);
        primaryStage.setScene(new Scene(root, 1700, 600));
        primaryStage.show();
    }
}
