package com.expleague.ml;

import javafx.application.Platform;
import javafx.scene.chart.XYChart;
import javafx.scene.control.ProgressBar;

public class BuildProgressHandler {
    private ProgressBar progressBar;
    private int stepsCount;
    private int curStep = 0;

    public BuildProgressHandler(ProgressBar progressBar) {
        this.progressBar = progressBar;
        this.stepsCount = 1;
        this.curStep = 0;
    }

    public BuildProgressHandler(ProgressBar progressBar, int stepsCount) {
        this.progressBar = progressBar;
        this.stepsCount = stepsCount;
        this.curStep = 0;
    }

    public void update(double progress) {
        if (progressBar == null) return;

        Platform.runLater(() -> {
            progressBar.setProgress(progress);
        });
    }

    public void step() {
        curStep++;
        if (progressBar == null) return;
        Platform.runLater(() -> {
            progressBar.setProgress((double)curStep / stepsCount);
        });

    }
}
