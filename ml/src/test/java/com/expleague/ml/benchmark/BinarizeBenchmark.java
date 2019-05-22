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
import javafx.scene.control.Label;
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

    private Label algorithm1BinTime = new Label();
    private Label algorithm2BinTime = new Label();
    private Label algorithm3BinTime = new Label();
    private Label algorithm4BinTime = new Label();

    private XYChart.Series barsUsageSeries1 = new XYChart.Series();
    private XYChart.Series barsUsageSeries2 = new XYChart.Series();
    private XYChart.Series barsUsageSeries3 = new XYChart.Series();
    private XYChart.Series barsUsageSeries4 = new XYChart.Series();

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

        algorithm1BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 0);
        algorithm2BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 1);
        algorithm3BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 2);
        algorithm4BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 3);

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
                            new BuildProgressHandler(algorithm1Bar, BFGridFactory.getStepsCount(MethodType.MEDIAN, dataset.vecData(), 32)),
                            0,
                            barsUsageSeries1,
                            algorithm1BinTime).run();
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
                            new BuildProgressHandler(algorithm2Bar, BFGridFactory.getStepsCount(MethodType.PROBABILITY_PRESORT, dataset.vecData(), 32)),
                            1,
                            barsUsageSeries2,
                            algorithm2BinTime).run();
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
                            new BuildProgressHandler(algorithm3Bar, BFGridFactory.getStepsCount(MethodType.PROBABILITY_PRESORT, dataset.vecData(), 32)),
                            2,
                            barsUsageSeries3,
                            algorithm3BinTime).run();
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
                            new BuildProgressHandler(algorithm4Bar, BFGridFactory.getStepsCount(MethodType.PROBABILITY_PRESORT, dataset.vecData(), 32)),
                            3,
                            barsUsageSeries4,
                            algorithm4BinTime).run();
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

        barsUsageSeries1.setName("Bars usage 1");
        barsUsageSeries2.setName("Bars usage 2");
        barsUsageSeries3.setName("Bars usage 3");
        barsUsageSeries4.setName("Bars usage 4");

        BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries1, 0);
        BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries2, 1);
        BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries3, 2);
        BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries4, 3);

        root.getChildren().add(gridpane);
        primaryStage.setScene(new Scene(root, 1920, 900));
        primaryStage.show();
    }
}
