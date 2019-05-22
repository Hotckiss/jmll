package com.expleague.ml;

import com.expleague.commons.FileTestCase;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.greedyRegion.GreedyProbLinearRegion;
import com.expleague.ml.testUtils.TestResourceLoader;
import gnu.trove.list.array.TIntArrayList;
import org.junit.Assert;

import java.io.IOException;
import java.math.BigInteger;
import java.util.List;
import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 12.11.12
 * Time: 16:35
 */
public class GridTest extends FileTestCase {
  public static Pool<?> learn, validate, all, all10;
  private static FastRandom rng = new FastRandom(0);

  private static synchronized void loadDataSet() {
    try {
      if (learn == null || validate == null) {
        learn = TestResourceLoader.loadPool("features.txt.gz");
        validate = TestResourceLoader.loadPool("featuresTest.txt.gz");
        all = TestResourceLoader.loadPool("featuresAll.txt");
        all10 = TestResourceLoader.loadPool("featuresAll10.txt");
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void testSplit() {
    List<? extends Pool<?>> split = DataTools.splitDataSet(all10, rng, 0.1, 0.9);
    System.out.println(split.size());
    Pool<?> p1 = split.get(0);
    //Pool<?> p2 = split.get(1);

    System.out.println(p1.vecData().xdim());
    //System.out.println(p2.vecData().xdim());
    System.out.println(p1.vecData().length());
    System.out.println(all10.vecData().length());
    //System.out.println(p2.vecData().length());
    //System.out.println(p1.vecData().data().get(10000, 35));

  }

  public void testFeaturesLoaded() {
    assertEquals(50, learn.vecData().xdim());
    assertEquals(50, validate.vecData().xdim());
    assertEquals(12465, learn.size());
    assertEquals(46596, validate.size());
  }

  public void testGrid1() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);

//    assertEquals(624, grid.size());
    checkResultByFile(BFGrid.CONVERTER.convertTo(grid).toString());
  }

  public void testBinary() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    assertEquals(1, grid.row(4).size());
  }

  public void testBinarize1() throws IOException {
     final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    //System.out.println(learn.vecData());
    assertEquals(1, grid.row(4).size());
    final byte[] bins = new byte[learn.vecData().xdim()];
    final Vec point = new ArrayVec(learn.vecData().xdim());
    point.set(0, 0.465441);
    point.set(17, 0);

    grid.binarizeTo(point, bins);

    //System.out.println(bins.length);
    assertEquals(28, bins[0]);
    assertEquals(0, bins[17]);
    assertFalse(grid.bf(28).value(bins));
    assertTrue(grid.bf(27).value(bins));
  }

  public void testBinarize2() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    assertEquals(1, grid.row(4).size());
    final byte[] bins = new byte[learn.vecData().xdim()];
    final Vec point = new ArrayVec(learn.vecData().xdim());
    point.set(0, 0.0);

    grid.binarizeTo(point, bins);
    assertEquals(0, bins[0]);
  }

  public void testBinarize3() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    assertEquals(1, grid.row(4).size());
    final byte[] bins = new byte[learn.vecData().xdim()];
    final Vec point = new ArrayVec(learn.vecData().xdim());
    point.set(3, 1.0);
    grid.binarizeTo(point, bins);
    final BFGrid.Feature bf = grid.bf(96);

    assertEquals(true, bf.value(bins));
    assertEquals(true, bf.value(point));
  }

  public void testSplitUniform() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(3, grid.size());
    assertEquals(0.1, grid.bf(0).condition());
    assertEquals(0.3, grid.bf(1).condition());
    assertEquals(0.6, grid.bf(2).condition());
  }

  public void testSplitUniformUnsorted() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0.8, 0.5, 0, 0.1, 0.2, 0.3, 0.6, 0.7));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(3, grid.size());
    assertEquals(0.1, grid.bf(0).condition());
    assertEquals(0.3, grid.bf(1).condition());
    assertEquals(0.6, grid.bf(2).condition());
  }

  public void testSplitBinary() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0, 1, 1, 1, 1, 1, 1));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(1, grid.size());
    assertEquals(0., grid.bf(0).condition());
  }

  public void testSplitBinaryIncorrect1() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(1, 1, 1, 1, 1, 1, 1, 1));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(0, grid.size());
  }

  public void testSplitBinaryIncorrect2() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 1, 1, 1, 1, 1, 1, 1));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(1, grid.size());
  }

  public void testBinarize4() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    final byte[] bin = new byte[1];
    grid.binarizeTo(new ArrayVec(0.), bin);
    assertEquals(0, bin[0]);
    grid.binarizeTo(new ArrayVec(1.), bin);
    assertEquals(3, bin[0]);
  }

  public void testSameFeatures() {
      final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0.0, 0.5, 0.3));
      final VecDataSet ds = new VecDataSetImpl(data, null);
      final BFGrid grid = GridTools.medianGrid(ds, 32);
      assertEquals(1, grid.size());
  }

  public void testIterativeP_cxVsFull() {
    final double lambda = 0.00001;
    final int n = 1000;
    final int c = 300;
    double D_prev = Double.NEGATIVE_INFINITY;
    double S_prev = Double.NEGATIVE_INFINITY;

    for (int k = 0; k < n; k++) {
      double D = 0;
      for (int i = c; i < n; i++) {
        D += Math.exp(-lambda * Math.abs(i - k));
      }
      double S = 0;
      for (int i = 0; i < n; i++) {
        S += Math.exp(-lambda * Math.abs(i - k));
      }
      if (Double.isFinite(D_prev)) {
        Assert.assertEquals(D, D_prev + Math.exp(-lambda * Math.abs(c - k)) - Math.exp(-lambda * (n - k - 1)), 0.01);
        Assert.assertEquals(S,S_prev + Math.exp(-lambda * k) - Math.exp(-lambda * (n - k - 1)), 0.01);
      }
      D_prev = D;
      S_prev = S;
    }
  }

  public void testProbLinearScoreFromLambda() {
    final VecDataSet ds = learn.vecData();
    final BFGrid grid = GridTools.medianGrid(ds, 32);
    final L2 target = learn.target(L2.class);
    WeightedLoss<L2> wloss = new WeightedLoss<>(target, IntStream.range(0, target.dim()).map(idx -> 1).toArray());
    BinarizedDataSet bds = ds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    GreedyProbLinearRegion.ScoreFromLambda scoreFromLambda = new GreedyProbLinearRegion.ScoreFromLambda(bds, IntStream.range(0, learn.size()).toArray(), wloss, true, grid.bf(10));

    Vec x = new ArrayVec(10, 10);
    Vec grad = new ArrayVec(2);
    scoreFromLambda.gradientTo(x, grad);
    Vec realGrad = new ArrayVec(2);
    double v = scoreFromLambda.value(x);
    x.adjust(0, MathTools.EPSILON);
    realGrad.set(0, (v - scoreFromLambda.value(x)) / MathTools.EPSILON);
    x.adjust(0, -MathTools.EPSILON);
    x.adjust(1, MathTools.EPSILON);
    realGrad.set(1, (v - scoreFromLambda.value(x)) / MathTools.EPSILON);
    x.adjust(1, -MathTools.EPSILON);
    assertTrue(VecTools.distanceL2(grad, realGrad) < 1e-3);
  }

  public void testPlotProbLinearScoreFromLambda() {
    final VecDataSet ds = learn.vecData();
    final BFGrid grid = GridTools.medianGrid(ds, 32);
    final L2 target = learn.target(L2.class);
    WeightedLoss<L2> wloss = new WeightedLoss<>(target, IntStream.range(0, target.dim()).map(idx -> 1).toArray());
    BinarizedDataSet bds = ds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    GreedyProbLinearRegion.ScoreFromLambda scoreFromLambda = new GreedyProbLinearRegion.ScoreFromLambda(bds, IntStream.range(0, learn.size()).toArray(), wloss, true, grid.bf(10));

    Vec x = new ArrayVec(10, 10);
    for (int i = 0; i < 100; i++) {
      x.set(0, i * 0.1 - 5);
      System.out.println(x + "\t" + scoreFromLambda.value(x));
    }
  }

  public void testInsertBorder() {
    TIntArrayList borders = new TIntArrayList();
    borders.add(new int[]{1, 3, 5, 7, 9});
    TIntArrayList mergedBorders = GridTools.insertBorder(borders, 2);
    assertEquals(2, mergedBorders.get(1));
    assertEquals(6, mergedBorders.size());
  }

  // feature with same sorted order
  public void testCalculatePartitionScore1() {
    int[] mapper = new int[]{0, 0, 0, 0, 1, 1, 1, 1};
    double[] feature2 = new double[]{1, 2, 3, 4, 5, 6, 7, 8};
    TIntArrayList bordersFeature2 = new TIntArrayList();
    bordersFeature2.add(8); // last border
    int newBorder = 2;
    double score1 = GridTools.calculatePartitionScore(mapper, feature2, bordersFeature2, newBorder);
    assertEquals(-8.317, score1, 0.001);
  }

  // feature with same sorted order
  public void testCalculatePartitionScore2() {
    int[] mapper = new int[]{0, 0, 0, 0, 1, 1, 1, 1};
    double[] feature2 = new double[]{1, 2, 3, 4, 5, 6, 7, 8};
    TIntArrayList bordersFeature2 = new TIntArrayList();
    bordersFeature2.add(8); // last border
    int newBorder = 3;
    double score1 = GridTools.calculatePartitionScore(mapper, feature2, bordersFeature2, newBorder);
    System.out.println(score1);
    assertEquals(-8.841, score1, 0.001);
  }

  // feature with same sorted order
  public void testCalculatePartitionScore3() {
    int[] mapper = new int[]{0, 0, 0, 0, 1, 1, 1, 1};
    double[] feature2 = new double[]{1, 2, 3, 4, 5, 6, 7, 8};
    TIntArrayList bordersFeature2 = new TIntArrayList();
    bordersFeature2.add(8); // last border
    int newBorder = 1;
    double score1 = GridTools.calculatePartitionScore(mapper, feature2, bordersFeature2, newBorder);
    assertEquals(-8.841, score1, 0.001);
  }

  // feature with same sorted order
  public void testCalculatePartitionScore4() {
    int[] mapper = new int[]{0, 0, 0, 0, 1, 1, 1, 1};
    double[] feature2 = new double[]{1, 2, 3, 4, 5, 6, 7, 8};
    TIntArrayList bordersFeature2 = new TIntArrayList();
    bordersFeature2.add(8); // last border
    int newBorder = 4;
    double score1 = GridTools.calculatePartitionScore(mapper, feature2, bordersFeature2, newBorder);
    assertEquals(-11.09, score1, 0.001);
  }

  // f1: 1234|5678 / f2: 37561284
  public void testCalculatePartitionScore5() {
    int[] mapper = new int[]{0, 1, 1, 1, 0, 0, 0, 1};
    double[] feature2 = new double[]{3, 7, 5, 6, 1, 2, 8, 4};
    TIntArrayList bordersFeature2 = new TIntArrayList();
    bordersFeature2.add(8); // last border
    int newBorder = 6;
    double score1 = GridTools.calculatePartitionScore(mapper, feature2, bordersFeature2, newBorder);
    assertEquals(-6.591, score1, 0.001);
  }

  // feature with same sorted order
  public void testBestPartition1() {
    int[] mapper = new int[]{0, 0, 0, 0, 1, 1, 1, 1};
    double[] feature2 = new double[]{1, 2, 3, 4, 5, 6, 7, 8};
    TIntArrayList bordersFeature2 = new TIntArrayList();
    bordersFeature2.add(8); // last border
    GridTools.PartitionResult result = GridTools.bestPartition(mapper, feature2, bordersFeature2);
    assertEquals(2, result.getSplitPosition());
  }

  // f1: 1234|5678 / f2: 37561284
  public void testBestPartition2() {
    int[] mapper = new int[]{0, 1, 1, 1, 0, 0, 0, 1};
    double[] feature2 = new double[]{3, 7, 5, 6, 1, 2, 8, 4};
    TIntArrayList bordersFeature2 = new TIntArrayList();
    bordersFeature2.add(8); // last border
    GridTools.PartitionResult result = GridTools.bestPartition(mapper, feature2, bordersFeature2);
    assertEquals(3, result.getSplitPosition());
  }

  public void testBuildBinsMapper1() {
    double[] f1 = new double[]{1, 2, 3, 4, 5, 6, 7, 8};
    double[] f2 = new double[]{5, 6, 1, 8, 3, 4, 2, 7};

    int[] sortedf1 = new int[]{0, 1, 2, 3, 4, 5, 6, 7};
    int[] sortedf2 = new int[]{2, 6, 4, 5, 0, 1, 7, 3};

    TIntArrayList bordersf1 = new TIntArrayList();
    bordersf1.add(4);
    bordersf1.add(8);

    int[] actual = GridTools.buildBinsMapper(bordersf1, sortedf1, sortedf2);
    int[] expected = new int[]{0, 1, 1, 1, 0, 0, 1, 0};

    for (int i = 0; i < actual.length; i++) {
      assertEquals(expected[i], actual[i]);
    }
  }

  public void testBuildBinsMapper2() {
    double[] f1 = new double[]{1, 2, 3, 4, 5, 6, 7, 8};
    double[] f2 = new double[]{5, 6, 1, 8, 3, 4, 2, 7};

    int[] sortedf1 = new int[]{0, 1, 2, 3, 4, 5, 6, 7};
    int[] sortedf2 = new int[]{0, 1, 2, 3, 4, 5, 6, 7};

    TIntArrayList bordersf1 = new TIntArrayList();
    bordersf1.add(4);
    bordersf1.add(8);

    int[] actual = GridTools.buildBinsMapper(bordersf1, sortedf1, sortedf2);
    int[] expected = new int[]{0, 0, 0, 0, 1, 1, 1, 1};

    for (int i = 0; i < actual.length; i++) {
      assertEquals(expected[i], actual[i]);
    }
  }

  public void testProbabilityBinarize1() {
    final VecBasedMx data = new VecBasedMx(2, new ArrayVec(1, 3, 2, 7, 3, 5, 4, 6, 5, 1, 6, 2, 7, 8, 8, 4));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 2, false);
  }

  public void testProbabilityBinarize2() {
    final VecBasedMx data = new VecBasedMx(2, new ArrayVec(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 2, true);
  }

  public void testProbabilityBinarize3() {
    final VecBasedMx data = new VecBasedMx(3, new ArrayVec(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 3, false);
  }

  public void testProbabilityBinarize4() {
    //final VecBasedMx data = new VecBasedMx(2, new ArrayVec(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8));
    //final VecDataSet ds = new VecDataSetImpl(data, null);
    //System.out.println(learn.vecData().xdim());
    //System.out.println(learn.vecData().length());
    BFGrid grid = GridTools.probabilityGrid(learn.vecData(), 32, true);
  }

  public void testProbabilityBinarize5() {
    final VecBasedMx data = new VecBasedMx(2, new ArrayVec(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 2, false);
  }

  public void testProbabilityBinarize6() {
    int xdim = 5;
    int len = 4096;
    double[] arr = new double[xdim * len];
    for (int i = 0; i < len; i++) {
      for (int j = 0; j < xdim; j++) {
        arr[xdim * i + j] = i;
      }
    }

    ArrayVec vec = new ArrayVec(arr, 0, xdim * len);
    final VecBasedMx data = new VecBasedMx(xdim, vec);
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 8, true);
  }

  public void testProbabilityBinarize7() {
    final VecBasedMx data = new VecBasedMx(2, new ArrayVec(1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 2, false);
    BFGrid grid1 = GridTools.probabilityGrid(ds, 2, true);
    BFGrid grid2 = GridTools.medianGrid(ds, 2);
  }

  public void testProbabilityBinarize8() {
    final VecBasedMx data = new VecBasedMx(2, new ArrayVec(1, 0, 2, 1, 4, 2, 8, 3, 16, 4, 32, 5, 64, 6, 128, 7));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 2, false);
    BFGrid grid1 = GridTools.probabilityGrid(ds, 2, true);
    BFGrid grid2 = GridTools.medianGrid(ds, 2);
  }

  public void testProbabilityBinarize9() {
    final VecBasedMx data = new VecBasedMx(2, new ArrayVec(1, 0, 2, 1, 4, 2, 8, 3, 16, 4, 32, 5, 64, 6, 128, 7, 256, 8, 512, 9, 1024, 10, 2048, 11, 4096, 12, 8192, 13, 16384, 14, 32768, 15, 65536, 16));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 2, false);
    BFGrid grid1 = GridTools.probabilityGrid(ds, 2, true);
    BFGrid grid2 = GridTools.medianGrid(ds, 2);
  }

  public void testProbabilityBinarize10() {
    final VecDataSet ds = learn.vecData();
    double[] feature = new double[ds.length()];
    for (int f = 0; f < ds.xdim(); f++) {
      //if (!(f == 15 || f==17 || f==24 || f==35)) continue;
      final ArrayPermutation permutation = new ArrayPermutation(ds.order(f));
      final int[] order = permutation.direct();
      final int[] reverse = permutation.reverse();

      for (int i = 0; i < feature.length; i++)
        feature[i] = ds.at(order[i]).get(f);

      //System.out.print("F: " + f + " ");
      int l = 10;
      int off = 12400;
      for (int j = feature.length - l - off; j < feature.length - off; j++) {
        System.out.print(order[j] + " ");
      }
      System.out.println();
    }
  }

  public void testProbabilityBinarizeCmp() {
    int xdim = 5;
    int len = 4096;
    double[] arr = new double[xdim * len];
    for (int i = 0; i < len; i++) {
      for (int j = 0; j < xdim; j++) {
        arr[xdim * i + j] = i;
      }
    }

    ArrayVec vec = new ArrayVec(arr, 0, xdim * len);
    final VecBasedMx data = new VecBasedMx(xdim, vec);
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid_bigInt(ds, 16);
    BFGrid grid1 = GridTools.probabilityGrid_bigInt(ds, 16);
  }

  public void testProbabilityBinarizeCmp2() {
    int xdim = 2;
    int len = 16;
    double[] arr = new double[xdim * len];
    for (int i = 0; i < len; i++) {
      for (int j = 0; j < xdim; j++) {
        arr[xdim * i + j] = i;
      }
    }

    ArrayVec vec = new ArrayVec(arr, 0, xdim * len);
    final VecBasedMx data = new VecBasedMx(xdim, vec);
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 2, true);
    //BFGrid grid1 = GridTools.probabilityGrid(ds, 1, true);
  }

  public void testProbabilityBinarizeCmp22() {
    int xdim = 50;
    int len = learn.vecData().length();

    double[] arr = new double[xdim * len];
    for (int i = 0; i < len; i++) {
      for (int j = 0; j < xdim; j++) {
        arr[xdim * i + j] = learn.vecData().data().get(i, j);
      }
    }

    ArrayVec vec = new ArrayVec(arr, 0, xdim * len);
    final VecBasedMx data = new VecBasedMx(xdim, vec);
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 8, true);
    //BFGrid grid1 = GridTools.probabilityGrid(ds, 1, true);
  }

  public void testProbabilityBinarizeCmp3() {
    int xdim = 5;
    int len = 4096;
    double[] arr = new double[xdim * len];
    for (int i = 0; i < len; i++) {
      for (int j = 0; j < xdim; j++) {
        arr[xdim * i + j] = i;
      }
    }

    ArrayVec vec = new ArrayVec(arr, 0, xdim * len);
    final VecBasedMx data = new VecBasedMx(xdim, vec);
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid_bigInt(ds, 1);
    BFGrid grid1 = GridTools.probabilityGrid_bigInt(ds, 1);
  }

  public void testBigInt() {
    BigInteger integer = new BigInteger("2", 10);
    integer = integer.multiply(new BigInteger("5", 10));
    System.out.println(integer);
  }

  public void testFirstPartition1() {
    TIntArrayList arr = new TIntArrayList();
    arr.add(2);

    assertEquals(1, GridTools.firstPartition(arr));
  }

  public void testFirstPartition2() {
    TIntArrayList arr = new TIntArrayList();
    arr.add(1);
    arr.add(2);
    assertEquals(3, GridTools.firstPartition(arr));
  }

  public void testFirstPartition3() {
    TIntArrayList arr = new TIntArrayList();
    arr.add(1);
    arr.add(2);
    arr.add(3);
    arr.add(4);
    arr.add(6);
    assertEquals(5, GridTools.firstPartition(arr));
  }

  public void testFirstPartition4() {
    TIntArrayList arr = new TIntArrayList();
    assertEquals(1, GridTools.firstPartition(arr));
  }

  public void testProbabilityBinarizeCmp7() {
    int xdim = 2;
    int len = learn.vecData().length();
    int featureIdx = 1;

    final double[] feature = new double[len];
    final ArrayPermutation permutation = new ArrayPermutation(learn.vecData().order(featureIdx));
    final int[] order = permutation.direct();
    final int[] reverse = permutation.reverse();

    for (int i = 0; i < feature.length; i++)
      feature[i] = learn.vecData().at(order[i]).get(featureIdx);

    double[] arr = new double[feature.length * 2];
    for (int i = 0; i < feature.length; i++) {
      arr[2 * i] = feature[i];
      if (feature[i] > 0) {
        arr[2 * i + 1] = Math.log(feature[i]);
      } else {
        arr[2 * i + 1] = 0;
      }

    }

    ArrayVec vec = new ArrayVec(arr, 0, xdim * len);
    final VecBasedMx data = new VecBasedMx(xdim, vec);
    final VecDataSet ds = new VecDataSetImpl(data, null);
    BFGrid grid = GridTools.probabilityGrid(ds, 8, true);
    BFGrid grid2 = GridTools.medianGrid(ds, 8);
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    loadDataSet();
  }

  @Override
  protected String getInputFileExtension() {
    return ".txt";
  }

  @Override
  protected String getResultFileExtension() {
    return ".txt";
  }

  @Override
  protected String getTestDataPath() {
    try {
      return TestResourceLoader.getFullPath("grid/");
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }
}
