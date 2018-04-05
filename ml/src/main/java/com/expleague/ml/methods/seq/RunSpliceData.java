package com.expleague.ml.methods.seq;

import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.io.codec.seq.Dictionary;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqArray;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.optimization.impl.SAGADescent;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.zip.GZIPInputStream;

public class RunSpliceData {
  private static final List<String> CLASSES = Arrays.asList("EI", "IE", "N");
  private static final int ALPHABET_SIZE = 15;
  private static final int BOOST_ITERS = 450;
  private static final double BOOST_STEP = 1;
  private static final int MAX_STATE_COUNT = 6;
  private static final double GRAD_STEP = 0.001;
  private static final FastRandom random = new FastRandom(239);
  private static final int THREAD_COUNT = 1;
  private static final int EPOCH_COUNT = 100;

  final List<Seq<Integer>> trainData = new ArrayList<>();
  final List<Mx> trainDataAsMx = new ArrayList<>();
  Vec trainTarget;

  final List<Seq<Integer>> testData = new ArrayList<>();
  final List<Mx> testDataAsMx = new ArrayList<>();
  Vec testTarget;

  final Alphabet<Integer> alphabet = new IntAlphabet(ALPHABET_SIZE);
  private Dictionary<Character> dictionary;

  public void loadData() throws IOException {
    System.out.println("Number of cores: " + Runtime.getRuntime().availableProcessors());
    System.out.println("Alphabet size: " + ALPHABET_SIZE);
    System.out.println("States count: " + MAX_STATE_COUNT);
    System.out.println("GradBoost step: " + BOOST_STEP);
    System.out.println("GradBoost iters: " + BOOST_ITERS);
    System.out.println("GradDesc step: " + GRAD_STEP);
    System.out.println("Grad epoch: " + EPOCH_COUNT);

    final List<CharSeq> data = new ArrayList<>();
    final TIntList classes = new TIntArrayList();
    final int[] classCount = new int[CLASSES.size()];

    new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(
        Paths.get("ml/src/splice.data.txt.gz").toFile())), StandardCharsets.UTF_8)
    ).lines().forEach(line -> {
      final String[] tokens = line.split(",");
      final int clazz = CLASSES.indexOf(tokens[0]);
      if (clazz == -1) {
        throw new IllegalStateException("Unknown class " + tokens[0]);
      }
      classes.add(clazz);
      classCount[clazz]++;
      data.add(new CharSeqArray(tokens[2].trim().toCharArray()));
    });

    List<Character> alpha = Arrays.asList('A', 'C', 'G', 'T');
    DictExpansion<Character> de = new DictExpansion<>(alpha, ALPHABET_SIZE);
    for (int i = 0; i < 50; i++) {
      data.forEach(de::accept);
    }

    dictionary = de.result();

    System.out.println("New dictionary: " + dictionary.alphabet().toString());

    int[] freqs = new int[ALPHABET_SIZE];
    int[] sum = new int[1];
    data.forEach(word -> {
      Seq<Integer> seq = dictionary.parse(word);
      for (int i =0; i < seq.length(); i++) {
        freqs[seq.at(i)]++;
        sum[0]++;
      }
    });
    Map<String, Double> map = new TreeMap<>();
    for (int i = 0; i < ALPHABET_SIZE; i++) {
      map.put(dictionary.condition(i).toString(), 1.0 * freqs[i] / sum[0]);
    }

    map.entrySet().stream().sorted(Map.Entry.comparingByValue(Collections.reverseOrder())).forEach(it -> System.out.printf("%s: %.5f, ", it.getKey(), it.getValue()));
    System.out.println();
    final int sampleCount = Arrays.stream(classCount).min().orElse(0);
    final int trainCount = sampleCount * 8 / 10;
    int[] trainClasses = new int[trainCount * CLASSES.size()];
    int[] testClasses = new int[(sampleCount - trainCount) * CLASSES.size()];
    for (int clazz = 0; clazz < CLASSES.size(); clazz++) {
      int cnt = 0;
      for (int i = 0; i < data.size(); i++) {
        if (classes.get(i) != clazz) {
          continue;
        }

        final Seq<Integer> seq = dictionary.parse(data.get(i));
        final Mx mx = new VecBasedMx(seq.length(), dictionary.size());
        for (int j = 0; j < seq.length(); j++) {
          mx.set(j, seq.at(j), 1);
        }
        if (cnt < trainCount) {
          trainClasses[trainData.size()] = classes.get(i);
          trainData.add(seq);
          trainDataAsMx.add(mx);
        } else if (cnt < sampleCount){
          testClasses[testData.size()] = classes.get(i);
          testData.add(seq);
          testDataAsMx.add(mx);
        }
        cnt++;
      }
    }

    trainTarget = VecTools.fromIntSeq(new IntSeq(trainClasses));
    testTarget = VecTools.fromIntSeq(new IntSeq(testClasses));
  }

  public void test() {

    DataSet<Seq<Integer>> data = new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return trainData.get(i);
      }

      @Override
      public int length() {
        return trainData.size();
      }

      @Override
      public Class<Seq<Integer>> elementType() {
        return null;
      }
    };

    List<Integer> labels = new ArrayList<>();
    for (int i = 0; i < trainTarget.length(); i++) {
      labels.add((int) Math.round(trainTarget.get(i)));
    }

    long start = System.nanoTime();
    final GradientSeqBoosting<Integer, LLLogit> boosting = new GradientSeqBoosting<>(
        new BootstrapSeqOptimization<>(
            new PNFARegressor<>(
                MAX_STATE_COUNT, 1, alphabet, 0, 0, 3, random,
                new SAGADescent(GRAD_STEP, 10000, random)
//                new AdamDescent(random, 100, 4),
            )
            , random
        ), BOOST_ITERS, BOOST_STEP
    );

    Consumer<Function<Seq<Integer>,Vec>> listener = classifier -> {
      System.out.println("Current time: " + new SimpleDateFormat("yyyyMMdd_HH:mm:ss").format(Calendar.getInstance().getTime()));
      System.out.println("Train loss: " + getLoss(trainData, trainTarget, classifier));
      System.out.println("Test loss: " + getLoss(testData, testTarget, classifier));
    };
    boosting.addListener(listener);

    final Function<Seq<Integer>, Vec> classifier =
        new OneVsRest(boosting, CLASSES.size(), labels).fit(data, new LLLogit(trainTarget, null));
    long end = System.nanoTime();

    System.out.println(String.format("Elapsed %.2f minutes", (end - start) / 60e9));
    System.out.println("Train accuracy of " + getAccuracy(trainData, trainTarget, classifier));
    System.out.println("Test accuracy of  " + getAccuracy(testData, testTarget, classifier));
  }

  /*
  @NotNull
  private GradientSeqBoosting<Integer, LLLogit> getBoosting() {
    return new GradientSeqBoosting<>(
          new BootstrapSeqOptimization<>(
              new PNFARegressor<>(MAX_STATE_COUNT, alphabet.size(), random, new SAGADescent(
                  GRAD_STEP, DESCENT_STEP_COUNT, random, THREAD_COUNT
              )), random
          ), BOOST_ITERS, BOOST_STEP
      );
  }

  @NotNull
  private GradientSeqBoosting<Integer, LLLogit> getBoostingGPU() {
    return new GradientSeqBoosting<>(
        new BootstrapSeqOptimization<>(
            new PNFAonGPU<>(MAX_STATE_COUNT, alphabet.size(), random, new SAGADescentGPU(
                GRAD_STEP, DESCENT_STEP_COUNT, random, THREAD_COUNT
            )), random
        ), BOOST_ITERS, BOOST_STEP
    );
  }*/

  private class OneVsRest implements SeqOptimization<Integer, LLLogit> {
    private final int classCount;
    private final SeqOptimization<Integer, LLLogit> optimization;
    private final List<Integer> labels;

    OneVsRest(SeqOptimization<Integer, LLLogit> optimization, int classCount, List<Integer> labels) {
      this.classCount = classCount;
      this.optimization = optimization;
      this.labels = labels;
    }

    @Override
    public Function<Seq<Integer>, Vec> fit(DataSet<Seq<Integer>> learn, LLLogit llLogit) {
      List<Function<Seq<Integer>,Vec>> classifiers = new ArrayList<>();
      for (int i = 0; i < classCount; i++) {
        Vec newTarget = VecTools.copy(llLogit.labels());
        for (int j = 0; j < llLogit.labels().length(); j++) {
          if (labels.get(j) != i) {
            newTarget.set(j, 0);
          } else {
            newTarget.set(j, 1);
          }
        }
        classifiers.add(optimization.fit(learn, new LLLogit(newTarget, null)));
      }

      return argument -> {
        double max = -Double.MAX_VALUE;
        int clazz = -1;
        for (int i = 0; i < classCount; i++) {
          double prob = classifiers.get(i).apply(argument).get(0);
          if (prob > max) {
            max = prob;
            clazz = i;
          }
        }
        return new SingleValueVec(clazz);
      };
    }
  }

  private double getLoss(List<Seq<Integer>> data, Vec target, Function<Seq<Integer>, Vec> classifier) {
    final LLLogit lllogit = new LLLogit(target, null);
    Vec values = new ArrayVec(target.dim());
    for (int i =0 ; i < target.dim(); i++) {
      values.set(i, classifier.apply(data.get(i)).get(0));
    }
    return lllogit.value(values);
  }


  private double getAccuracy(List<Seq<Integer>> data, Vec target, Function<Seq<Integer>, Vec> classifier) {
    int passedCnt = 0;
    int classAccuracy[] = new int[3];
    int classSize[] = new int[3];
    for (int i = 0; i < data.size(); i++) {
      if (Math.round(classifier.apply(data.get(i)).get(0)) == target.get(i)) {
        passedCnt++;
        classAccuracy[(int) Math.round(target.get(i))]++;
      }
      classSize[(int) Math.round(target.get(i))]++;
    }
    for (int i = 0; i < 3; i++) {
      System.out.println("Class " + i + " accuracy:" + classAccuracy[i] * 1.0 / classSize[i]);
    }
    return 1.0 * passedCnt / data.size();
  }

  public static void main(String[] args) throws IOException {
    RunSpliceData test = new RunSpliceData();
    test.loadData();
    test.test();
  }
}