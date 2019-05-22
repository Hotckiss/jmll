package com.expleague.ml.benchmark.calcers;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.func.Ensemble;
import javafx.application.Platform;
import javafx.scene.chart.XYChart;

import java.io.PrintWriter;

public class BenchmarkValidateScoreCalcer implements ProgressHandler {
    private final String message;
    private final Vec current;
    private final VecDataSet ds;
    private final TargetFunc target;
    private final PrintWriter printWriter;
    private int index = 0;
    private XYChart.Series series;

    public BenchmarkValidateScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target, final PrintWriter printWriter, XYChart.Series series) {
        this.message = message;
        this.ds = ds;
        this.target = target;
        this.printWriter = printWriter;
        this.series = series;
        current = new ArrayVec(ds.length());
    }

    private double min = 1e10;

    @Override
    public void accept(final Trans partial) {
        index++;
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
        Platform.runLater(() -> {
            series.getData().add(new XYChart.Data(index, min));
        });

    }
}
