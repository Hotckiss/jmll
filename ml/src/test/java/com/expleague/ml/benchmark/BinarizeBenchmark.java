package com.expleague.ml.benchmark;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.benchmark.generators.FakePoolsGenerator;
import com.expleague.ml.benchmark.ml.*;
import com.expleague.ml.benchmark.ui.BinarizeBenchmarkUIUtils;
import com.expleague.ml.benchmark.utils.SettingsConfig;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.testUtils.TestResourceLoader;
import javafx.application.Application;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.HPos;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.ArrayList;

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

    private ComboBox box1 = new ComboBox();
    private ComboBox box2 = new ComboBox();
    private ComboBox box3 = new ComboBox();
    private ComboBox box4 = new ComboBox();

    private MethodType method1 = MethodType.MEDIAN;
    private MethodType method2 = MethodType.MEDIAN;
    private MethodType method3 = MethodType.MEDIAN;
    private MethodType method4 = MethodType.MEDIAN;

    private LineChart<Number,Number> algorithm1Chart = null;
    private LineChart<Number,Number> algorithm2Chart = null;
    private LineChart<Number,Number> algorithm3Chart = null;
    private LineChart<Number,Number> algorithm4Chart = null;

    private BarChart bar1Chart = null;
    private BarChart bar2Chart = null;
    private BarChart bar3Chart = null;
    private BarChart bar4Chart = null;

    private Label algorithm1FinalScore = new Label();
    private Label algorithm2FinalScore = new Label();
    private Label algorithm3FinalScore = new Label();
    private Label algorithm4FinalScore = new Label();

    private Label algorithm1TotalBins = new Label();
    private Label algorithm2TotalBins = new Label();
    private Label algorithm3TotalBins = new Label();
    private Label algorithm4TotalBins = new Label();

    private CheckBox thread1Enabled = new CheckBox();
    private CheckBox thread2Enabled = new CheckBox();
    private CheckBox thread3Enabled = new CheckBox();
    private CheckBox thread4Enabled = new CheckBox();

    private Button settingsButton = new Button();

    private SettingsConfig settings = new SettingsConfig();

    private Label rs = new Label();
    private Label ls = new Label();
    private Label bf = new Label();
    private Label td = new Label();
    private Label ic = new Label();
    private Label ss = new Label();

    private ComboBox datasetSelection = new ComboBox();
    private DatasetType currentType = DatasetType.FEATURES_TXT;

    private static synchronized void loadDataSet() {
        try {
            dataset = DatasetsFactory.makePool(DatasetType.FEATURES_TXT);
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

        Label datasetLabel = new Label("Dataset path:");
        datasetLabel.setStyle
                (
                        "-fx-font-size: 11px;"
                                + "-fx-font-weight: 500;"
                );
        TextField datasetInput = new TextField();
        Button datasetLoadButton = new Button(" Load ");
        datasetLoadButton.setStyle
                (
                        "-fx-font-size: 11px;"
                                + "-fx-background-color: #c2d22b;"
                                + "-fx-border-style: solid inside;"
                                + "-fx-border-width: 0pt;"
                );
        BinarizeBenchmarkUIUtils.addDatasetInput(gridpane, datasetLabel, datasetInput, datasetLoadButton);

        Label featuresLabel = new Label("Features columns:");
        featuresLabel.setStyle
                (
                        "-fx-font-size: 11px;"
                                + "-fx-font-weight: 500;"
                );
        TextField featuresInput = new TextField();
        Button featuresApplyButton = new Button("Apply");
        featuresApplyButton.setStyle
                (
                        "-fx-font-size: 11px;"
                                + "-fx-background-color: #c2d22b;"
                                + "-fx-border-style: solid inside;"
                                + "-fx-border-width: 0pt;"
                );
        BinarizeBenchmarkUIUtils.addFeaturesInput(gridpane, featuresLabel, featuresInput, featuresApplyButton);

        Label targetColumnLabel = new Label("Target column:");
        targetColumnLabel.setStyle
                (
                        "-fx-font-size: 11px;"
                                + "-fx-font-weight: 500;"
                );
        TextField targetInput = new TextField();
        Button targetApplyButton = new Button("Apply");
        targetApplyButton.setStyle
                (
                        "-fx-font-size: 11px;"
                                + "-fx-background-color: #c2d22b;"
                                + "-fx-border-style: solid inside;"
                                + "-fx-border-width: 0pt;"
                );
        BinarizeBenchmarkUIUtils.addTargetInput(gridpane, targetColumnLabel, targetInput, targetApplyButton);

        setupBinarizeProgress(gridpane);

        Button runButt = new Button("Run!");
        GridPane.setHalignment(runButt, HPos.CENTER);
        gridpane.add(runButt, 7, 4, 2, 1);
        setupRunButton(runButt);

        setupScoreCharts(gridpane);
        setupBinsUsageCharts(gridpane);

        datasetSelection = BinarizeBenchmarkUIUtils.addDatasetSelection(gridpane);
        datasetSelection.valueProperty().addListener((ChangeListener<String>) (ov, t, t1) -> {
            currentType = DatasetsFactory.getDatasetType(t1);
            try {
                dataset = DatasetsFactory.makePool(currentType);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        settingsButton = BinarizeBenchmarkUIUtils.addSettingsInput(gridpane);

        settingsButton.setOnAction(event -> {
            GridPane settingsPane = new GridPane();
            settingsPane.setPadding(new Insets(5));
            ColumnConstraints column1 = new ColumnConstraints(140);
            ColumnConstraints column2 = new ColumnConstraints(190);
            settingsPane.getColumnConstraints().addAll(column1, column2);
            settingsPane.setHgap(8);
            settingsPane.setVgap(8);

            TextField randomSeedField = addSettingsItem(settingsPane, "Random seed:", String.valueOf(settings.getSeed()), 0);
            TextField learnSize = addSettingsItem(settingsPane, "Learn size:", String.valueOf(settings.getLearnSize()), 1);
            TextField binFactor = addSettingsItem(settingsPane, "Bin factor:", String.valueOf(settings.getBinFactor()), 2);
            TextField treeDepth = addSettingsItem(settingsPane, "Tree depth:", String.valueOf(settings.getTreeDepth()), 3);
            TextField itersCount = addSettingsItem(settingsPane, "Iterations:", String.valueOf(settings.getIters()), 4);
            TextField stepSize = addSettingsItem(settingsPane, "Step size:", String.valueOf(settings.getStep()), 5);


            Button apply = new Button("Apply");
            GridPane.setHalignment(apply, HPos.CENTER);
            settingsPane.add(apply, 0, 7, 2, 1);
            apply.setStyle
                    (
                            "-fx-font-size: 19px;"
                                    + "-fx-font-weight: bold;"
                                    + "-fx-background-color: lightgreen;"
                                    + "-fx-border-style: solid inside;"
                                    + "-fx-border-width: 0pt;"
                                    + "-fx-background-radius: 19pt; "
                    );
            Scene secondScene = new Scene(settingsPane, 360, 280);
            final Stage settingsWindow = new Stage();

            apply.setOnAction(event1 -> {
                try {
                    settings.setSeed(Integer.parseInt(randomSeedField.getText()));
                    settings.setLearnSize(Double.parseDouble(learnSize.getText()));
                    settings.setBinFactor(Integer.parseInt(binFactor.getText()));
                    settings.setTreeDepth(Integer.parseInt(treeDepth.getText()));
                    settings.setIters(Integer.parseInt(itersCount.getText()));
                    settings.setStep(Double.parseDouble(stepSize.getText()));
                } catch (NumberFormatException ex) {
                    //reset to default
                    settings = new SettingsConfig();
                }

                rs.setText(String.valueOf(settings.getSeed()));
                ls.setText(String.valueOf(settings.getLearnSize()));
                bf.setText(String.valueOf(settings.getBinFactor()));
                td.setText(String.valueOf(settings.getTreeDepth()));
                ic.setText(String.valueOf(settings.getIters()));
                ss.setText(String.valueOf(settings.getStep()));

                settingsWindow.close();
            });

            settingsWindow.setTitle("Settings");
            settingsWindow.setScene(secondScene);
            settingsWindow.show();
        });

        Label cfgTitle = new Label("current configuration".toUpperCase());
        cfgTitle.setStyle
                (
                        "-fx-font-size: 19px;"
                                + "-fx-font-weight: bold;"
                );
        GridPane.setHalignment(cfgTitle, HPos.CENTER);
        gridpane.add(cfgTitle, 1, 0, 4, 1);

        rs = addParam(gridpane, "Random seed:", String.valueOf(settings.getSeed()), 1, 1);
        ls = addParam(gridpane, "Learn size:", String.valueOf(settings.getLearnSize()), 2, 1);
        bf = addParam(gridpane, "Bin factor:", String.valueOf(settings.getBinFactor()), 3, 1);
        td = addParam(gridpane, "Tree depth:", String.valueOf(settings.getTreeDepth()), 1, 3);
        ic = addParam(gridpane, "Iterations:", String.valueOf(settings.getIters()), 2, 3);
        ss = addParam(gridpane, "Step size:", String.valueOf(settings.getStep()), 3, 3);

        root.getChildren().add(gridpane);
        primaryStage.setScene(new Scene(root, 1740, 960));
        primaryStage.show();
    }

    private Label addParam(GridPane gridPane, String name, String value, int row, int col) {
        Label label = new Label(name);
        label.setStyle
                (
                        "-fx-font-size: 15px;"
                                + "-fx-font-weight: 500;"
                );
        GridPane.setHalignment(label, HPos.LEFT);
        gridPane.add(label, col, row);

        Label labelV = new Label(value);
        labelV.setStyle
                (
                        "-fx-font-size: 15px;"
                                + "-fx-font-weight: bold;"
                );
        GridPane.setHalignment(labelV, HPos.CENTER);
        gridPane.add(labelV, col + 1, row);

        return labelV;
    }

    private TextField addSettingsItem(GridPane settingsPane, String title, String value, int index) {
        Label label = new Label(title);
        GridPane.setHalignment(label, HPos.LEFT);
        settingsPane.add(label, 0, index);
        TextField field = new TextField(value);
        GridPane.setHalignment(field, HPos.RIGHT);
        settingsPane.add(field, 1, index, 2, 1);

        return field;
    }

    private void setupScoreCharts(GridPane gridpane) {
        series1.setName("Algorithm 1");
        series2.setName("Algorithm 2");
        series3.setName("Algorithm 3");
        series4.setName("Algorithm 4");
        series1T.setName("Algorithm 1 train");
        series2T.setName("Algorithm 2 train");
        series3T.setName("Algorithm 3 train");
        series4T.setName("Algorithm 4 train");
        algorithm1Chart = BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series1, series1T, 0);
        algorithm2Chart = BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series2, series2T, 1);
        algorithm3Chart = BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series3, series3T, 2);
        algorithm4Chart = BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series4, series4T, 3);
        algorithm1FinalScore = BinarizeBenchmarkUIUtils.addScore(gridpane, 0);
        algorithm2FinalScore = BinarizeBenchmarkUIUtils.addScore(gridpane, 1);
        algorithm3FinalScore = BinarizeBenchmarkUIUtils.addScore(gridpane, 2);
        algorithm4FinalScore = BinarizeBenchmarkUIUtils.addScore(gridpane, 3);
    }

    private void setupBinsUsageCharts(GridPane gridpane) {
        barsUsageSeries1.setName("Bins usage 1");
        barsUsageSeries2.setName("Bins usage 2");
        barsUsageSeries3.setName("Bins usage 3");
        barsUsageSeries4.setName("Bins usage 4");
        bar1Chart = BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries1, 0);
        bar2Chart = BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries2, 1);
        bar3Chart = BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries3, 2);
        bar4Chart = BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries4, 3);
        box1 = BinarizeBenchmarkUIUtils.addAlgorithmSelection(gridpane, 0);
        box2 = BinarizeBenchmarkUIUtils.addAlgorithmSelection(gridpane, 1);
        box3 = BinarizeBenchmarkUIUtils.addAlgorithmSelection(gridpane, 2);
        box4 = BinarizeBenchmarkUIUtils.addAlgorithmSelection(gridpane, 3);
        box1.setOnAction(event -> method1 = BFGridFactory.getAlgorithmType(box1.getValue().toString()));
        box2.setOnAction(event -> method2 = BFGridFactory.getAlgorithmType(box2.getValue().toString()));
        box3.setOnAction(event -> method3 = BFGridFactory.getAlgorithmType(box3.getValue().toString()));
        box4.setOnAction(event -> method4 = BFGridFactory.getAlgorithmType(box4.getValue().toString()));
        algorithm1TotalBins = BinarizeBenchmarkUIUtils.addBinsCount(gridpane, 0);
        algorithm2TotalBins = BinarizeBenchmarkUIUtils.addBinsCount(gridpane, 1);
        algorithm3TotalBins = BinarizeBenchmarkUIUtils.addBinsCount(gridpane, 2);
        algorithm4TotalBins = BinarizeBenchmarkUIUtils.addBinsCount(gridpane, 3);
    }

    private void setupBinarizeProgress(GridPane gridpane) {
        algorithm1Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 0);
        algorithm2Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 1);
        algorithm3Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 2);
        algorithm4Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 3);

        algorithm1BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 0);
        algorithm2BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 1);
        algorithm3BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 2);
        algorithm4BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 3);

        thread1Enabled = BinarizeBenchmarkUIUtils.addAlgorithmEnabled(gridpane, 0);
        thread2Enabled = BinarizeBenchmarkUIUtils.addAlgorithmEnabled(gridpane, 1);
        thread3Enabled = BinarizeBenchmarkUIUtils.addAlgorithmEnabled(gridpane, 2);
        thread4Enabled = BinarizeBenchmarkUIUtils.addAlgorithmEnabled(gridpane, 3);
    }

    private void setupRunButton(Button runButt) {
        runButt.setStyle
                (
                        "-fx-font-size: 24px;"
                                + "-fx-font-weight: bold;"
                                + "-fx-background-color: lightgreen;"
                                + "-fx-border-style: solid inside;"
                                + "-fx-border-width: 0pt;"
                                + "-fx-background-radius: 24pt; "
                );

        runButt.setOnAction(event -> {
            if (thread1Enabled.isSelected()) {
                final String name1 = BFGridFactory.getAlgorithmName(method1);
                algorithm1Chart.setTitle(name1 + " score");
                series1.setName(name1 + " validate");
                series1T.setName(name1 + " train");
                barsUsageSeries1.setName(name1 + " bins usage");
                bar1Chart.setTitle(name1 + " bins usage");
                new Thread(() -> {
                    try {
                        new MethodRunner("Algorithm1Log.txt",
                                dataset,
                                new FastRandom(settings.getSeed()),
                                method1,
                                settings.getLearnSize(),
                                series1,
                                series1T,
                                settings.getBinFactor(),
                                settings.getTreeDepth(),
                                settings.getIters(),
                                settings.getStep(),
                                new BuildProgressHandler(algorithm1Bar, BFGridFactory.getStepsCount(method1, dataset.vecData(), settings.getBinFactor())),
                                0,
                                barsUsageSeries1,
                                algorithm1BinTime,
                                algorithm1FinalScore,
                                algorithm1TotalBins).run();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();
            }

            if(thread2Enabled.isSelected()) {
                final String name2 = BFGridFactory.getAlgorithmName(method2);
                algorithm2Chart.setTitle(name2 + " score");
                series2.setName(name2 + " validate");
                series2T.setName(name2 + " train");
                barsUsageSeries2.setName(name2 + " bins usage");
                bar2Chart.setTitle(name2 + " bins usage");
                new Thread(() -> {
                    try {
                        new MethodRunner("Algorithm2Log.txt",
                                dataset,
                                new FastRandom(settings.getSeed()),
                                method2,
                                settings.getLearnSize(),
                                series2,
                                series2T,
                                settings.getBinFactor(),
                                settings.getTreeDepth(),
                                settings.getIters(),
                                settings.getStep(),
                                new BuildProgressHandler(algorithm2Bar, BFGridFactory.getStepsCount(method2, dataset.vecData(), settings.getBinFactor())),
                                1,
                                barsUsageSeries2,
                                algorithm2BinTime,
                                algorithm2FinalScore,
                                algorithm2TotalBins).run();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();
            }

            if (thread3Enabled.isSelected()) {
                final String name3 = BFGridFactory.getAlgorithmName(method3);
                algorithm3Chart.setTitle(name3 + " score");
                series3.setName(name3 + " validate");
                series3T.setName(name3 + " train");
                barsUsageSeries3.setName(name3 + " bins usage");
                bar3Chart.setTitle(name3 + " bins usage");
                new Thread(() -> {
                    try {
                        new MethodRunner("Algorithm3Log.txt",
                                dataset,
                                new FastRandom(settings.getSeed()),
                                method3,
                                settings.getLearnSize(),
                                series3,
                                series3T,
                                settings.getBinFactor(),
                                settings.getTreeDepth(),
                                settings.getIters(),
                                settings.getStep(),
                                new BuildProgressHandler(algorithm3Bar, BFGridFactory.getStepsCount(method3, dataset.vecData(), settings.getBinFactor())),
                                2,
                                barsUsageSeries3,
                                algorithm3BinTime, algorithm3FinalScore, algorithm3TotalBins).run();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();
            }

            if (thread4Enabled.isSelected()) {
                final String name4 = BFGridFactory.getAlgorithmName(method4);
                algorithm4Chart.setTitle(name4 + " score");
                series4.setName(name4 + " validate");
                series4T.setName(name4 + " train");
                barsUsageSeries4.setName(name4 + " bins usage");
                bar4Chart.setTitle(name4 + " bins usage");
                new Thread(() -> {
                    try {
                        new MethodRunner("Algorithm4Log.txt",
                                dataset,
                                new FastRandom(settings.getSeed()),
                                method4,
                                settings.getLearnSize(),
                                series4,
                                series4T,
                                settings.getBinFactor(),
                                settings.getTreeDepth(),
                                settings.getIters(),
                                settings.getStep(),
                                new BuildProgressHandler(algorithm4Bar, BFGridFactory.getStepsCount(method4, dataset.vecData(), settings.getBinFactor())),
                                3,
                                barsUsageSeries4,
                                algorithm4BinTime, algorithm4FinalScore, algorithm4TotalBins).run();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();
            }
        });
    }
}
