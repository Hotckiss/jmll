package com.expleague.exp.multiclass;

import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.GridTools;
import com.expleague.ml.cli.output.printers.MulticlassProgressPrinter;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.factorization.impl.ElasticNetFactorization;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.LogL2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.MultiClass;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.multiclass.gradfac.GradFacMulticlass;
import com.expleague.ml.methods.multiclass.gradfac.MultiClassColumnBootstrapOptimization;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.MultiClassModel;
import junit.framework.TestCase;

import java.io.IOException;

/**
 * User: qdeee
 * Date: 24.05.15
 */
public class DiplomaGradFacTest extends TestCase{
  private static Pool<?> learn;
  private static Pool<?> test;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    init();
  }

  private synchronized static void init() throws IOException {
    if (learn == null || test == null) {
      learn = DataTools.loadFromFeaturesTxt("/Users/qdeee/datasets/letter.tsv.learn");
      test = DataTools.loadFromFeaturesTxt("/Users/qdeee/datasets/letter.tsv.test");
    }
  }

  public void testBaseline() throws Exception {
    final MultiClass learner = new MultiClass(
        new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
        LogL2.class
    );
    fitModel(learner, 400, 0.3);
  }

  public void testGradFacElasticNet() throws Exception {
    final GradFacMulticlass learner = new GradFacMulticlass(
        new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
        new ElasticNetFactorization(20, 1e-2, 0.95, 0.15 * 1e-6),
        LogL2.class
    );
    fitModel(learner, 7500, 7.);
  }

  public void testGradFacElasticNetColumnsBootstrap() throws Exception {
    final MultiClassColumnBootstrapOptimization learner = new MultiClassColumnBootstrapOptimization(
        new GradFacMulticlass(
            new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
            new ElasticNetFactorization(20, 1e-2, 0.95, 0.15 * 1e-6),
            LogL2.class
        ),
        new FastRandom(100500),
        1.
    );
    fitModel(learner, 5000, 7.);
  }

  private void fitModel(final VecOptimization<L2> weak, final int iters, final double step) {
    final VecDataSet vecDataSet = learn.vecData();
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final MulticlassProgressPrinter multiclassProgressPrinter = new MulticlassProgressPrinter(learn, test);

    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(weak, L2.class, iters, step);
    boosting.addListener(multiclassProgressPrinter);
    final Ensemble ensemble = boosting.fit(vecDataSet, globalLoss);

    final MultiClassModel multiclassModel;
    if (ensemble.last() instanceof FuncJoin) {
      final FuncJoin joined = MCTools.joinBoostingResult(ensemble);
      multiclassModel = new MultiClassModel(joined);
    }
    else
      multiclassModel = new MultiClassModel(ensemble);

    Interval.start();
    System.out.println(MCTools.evalModel(multiclassModel, learn, "[LEARN] ", false));
    System.out.println(MCTools.evalModel(multiclassModel, test, "[TEST] ", false));
    Interval.stopAndPrint();
  }
}
