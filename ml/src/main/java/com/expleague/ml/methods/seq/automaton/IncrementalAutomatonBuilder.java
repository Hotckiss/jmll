package com.expleague.ml.methods.seq.automaton;

import com.expleague.commons.math.MathTools;
import com.expleague.ml.methods.seq.automaton.transform.*;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.SeqOptimization;

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

public class IncrementalAutomatonBuilder<T, Loss extends L2> implements SeqOptimization<T, Loss> {
  private final int maxStateCount;
  private final Alphabet<T> alphabet;
  private final Function<AutomatonStats<T>, Double> stateEvaluation;
  private final int maxIterations;
  private final ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);

  public IncrementalAutomatonBuilder(final Alphabet<T> alphabet,
                                     final Function<AutomatonStats<T>, Double> stateEvaluation,
                                     final int maxStateCount,
                                     final int maxIterations) {
    this.alphabet = alphabet;
    this.stateEvaluation = stateEvaluation;
    this.maxStateCount = maxStateCount;
    this.maxIterations = maxIterations;
  }

  @Override
  public Function<Seq<T>, Vec> fit(final DataSet<Seq<T>> learn, final Loss loss) {
    AutomatonStats<T> automatonStats = new AutomatonStats<>(alphabet, learn, loss);

    double oldCost = stateEvaluation.apply(automatonStats);

    for (int iter = 0; iter < maxIterations; iter++) {

      final AutomatonStats<T> automatonStats1 = automatonStats;
      final List<Future<AutomatonStats<T>>> futures = new ArrayList<>();
      for (Transform<T> transform: getTransforms(automatonStats)) {
        futures.add(executorService.submit(() -> transform.applyTransform(automatonStats1)));

        /*
        final AutomatonStats<T> newAutomatonStats = transform.applyTransform(automatonStats);
        final double newCost = stateEvaluation.compute(newAutomatonStats);
        if (newCost < optCost) {
          optCost = newCost;
          optTransform = transform;
        }*/
      }

      final AutomatonStats<T> optNewStats = futures.stream().map(future -> {
        try {
          return future.get();
        } catch (InterruptedException | ExecutionException e) {
          e.printStackTrace();
          return null;
        }
      }).min(Comparator.comparingDouble(stateEvaluation::apply)).orElse(null);
      final double optCost = stateEvaluation.apply(optNewStats);

      if (optNewStats == null || (optCost >= oldCost - 1e-9)) {
        System.out.println("Elapsed " + iter + " iterations");
        break;
      }

      automatonStats = optNewStats ;
      removeUnreachableStates(automatonStats);
      if (iter % 100 == 0 && iter != 0) {
        System.out.printf("Iter=%d, newCost=%f, state count=%d\n",
                iter, optCost, automatonStats.getAutomaton().getStateCount());
/*
        System.out.printf("Iter=%d, transform=%s, newCost=%f, state count=%d\n",
                iter, optTransform.getDescription(), optCost, automatonStats.getAutomaton().getStateCount());
                */
        System.out.flush();
      }
      oldCost = optCost;
    }

    final DFA<T> automaton = automatonStats.getAutomaton();
    final double[] stateValue = new double[automaton.getStateCount()];
    for (int i = 0; i < automaton.getStateCount(); i++) {
      if (automatonStats.getStateWeight().get(i) > MathTools.EPSILON) {
        stateValue[i] = automatonStats.getStateSum().get(i) / automatonStats.getStateWeight().get(i);
      }
    }
    System.out.println("Cur cost = " + stateEvaluation.apply(automatonStats));
    return argument -> new SingleValueVec(stateValue[automaton.run(argument)]);
  }


  private List<Transform<T>> getTransforms(final AutomatonStats<T> automatonStats) {
    final DFA<T> automaton = automatonStats.getAutomaton();
    final int stateCount = automaton.getStateCount();
    final Alphabet<T> alphabet = automatonStats.getAlphabet();
    final List<Transform<T>> transforms = new ArrayList<>();

    for (int from = 0; from < stateCount; from++) {
      for (int c = 0; c < alphabet.size(); c++) {
        if (automaton.hasTransition(from, alphabet.getT(alphabet.condition(c)))) {
          // todo commented out to improve performance
          transforms.add(new RemoveTransitionTransform<>(from, alphabet.getT(alphabet.condition(c))));
          for (int to = 0; to < stateCount; to++) {
            if (to != from) {
              // todo commented out to improve performance
                transforms.add(new ReplaceTransitionTransform<>(from, to, alphabet.getT(alphabet.condition(c))));
            }
          }
        } else {
          final T cT = alphabet.getT(alphabet.condition(c));
          if (stateCount < maxStateCount) {
            transforms.add(new SplitStateTransform<>(from, cT));
          }
          for (int to = 0; to < stateCount; to++) {
            transforms.add(new AddTransitionTransform<>(from, to, cT));
          }
        }
      }
      if (stateCount < maxStateCount) {
        for (int to = 0; to < stateCount; to++) {
          for (int c = 0; c < alphabet.size(); c++) {
            final T cT = alphabet.getT(alphabet.condition(c));
            if (!automaton.hasTransition(from, cT)) {
              for (int c1 = 0; c1 < alphabet.size(); c1++) {
                transforms.add(new AddNewStateTransform<>(from, to, cT, alphabet.getT(alphabet.condition(c1))));
              }
            }
          }
        }
      }
    }

    return transforms;
  }

  private void removeUnreachableStates(final AutomatonStats<T> automatonStats) {
    final DFA<T> automaton = automatonStats.getAutomaton();
    final Queue<Integer> queue = new LinkedList<>();
    final Alphabet<T> alphabet = automatonStats.getAlphabet();
    queue.add(automaton.getStartState());
    final boolean[] reached = new boolean[automaton.getStateCount()];
    reached[automaton.getStartState()] = true;

    while (!queue.isEmpty()) {
      final int v = queue.poll();
      for (int c = 0; c < automatonStats.getAlphabet().size(); c++) {
        final int to = automaton.getTransition(v, alphabet.getT(alphabet.condition(c)));
        if (to != -1 && !reached[to]) {
          queue.add(to);
          reached[to] = true;
        }
      }
    }
    for (int i = automaton.getStateCount() - 1; i >= 0; i--) {
      if (!reached[i] && i != automaton.getStartState()) {
        automaton.removeState(i);
        automatonStats.getSamplesEndState().remove(i);
        automatonStats.getStateWeight().remove(i);
        automatonStats.getStateSum().remove(i);
        automatonStats.getStateSum2().remove(i);
        automatonStats.getSamplesViaState().remove(i);
      }
    }
  }
}
