package com.spbsu.direct.gen;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.io.Vec2CharSequenceConverter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.*;
import com.spbsu.commons.util.ArrayTools;
import gnu.trove.list.TIntList;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;

import java.io.IOException;
import java.io.Writer;

import static com.spbsu.commons.math.vectors.VecTools.l1;
import static java.lang.Double.max;
import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 12.11.15
 * Time: 11:33
 */
public class SimpleGenerativeModel {
  public static final String EMPTY_ID = "##EMPTY##";
  public static final int GIBBS_COUNT = 10;

  private final WordGenProbabilityProvider[] providers;
  private final Dictionary<CharSeq> dict;
  private final FastRandom rng = new FastRandom(0);

  public double totalFreq = 0;
  public final TIntList freqs;

  public SimpleGenerativeModel(final Dictionary<CharSeq> dict,
                               final TIntList freqsLA) {
    this.dict = dict;
    this.providers = new WordGenProbabilityProvider[dict.size() + 1]; // +1 -- for EMPTY word

    for (int i = 0; i < providers.length; ++i) {
      this.providers[i] = new WordGenProbabilityProvider(i, dict);
    }

    this.freqs = freqsLA;
    this.totalFreq = freqsLA.sum();
  }

  /*
  public void loadStatistics(final String fileName) throws IOException {
    for (int i = 0; i < providers.length; ++i) {
      providers[i] = new WordGenProbabilityProvider(dict);
    }

    final Vec2CharSequenceConverter converter = new Vec2CharSequenceConverter();

    CharSeqTools.processLines(StreamTools.openTextFile(fileName), (Action<CharSequence>) sequence -> {
      final CharSequence[] split = CharSeqTools.split(sequence, '\t');

      final WordGenProbabilityProvider provider;
      if (!split[0].equals(EMPTY_ID)) {
        final CharSequence[] parts = CharSeqTools.split(split[0].subSequence(1, split[0].length() - 1), ", ");
        final SeqBuilder<CharSeq> builder = new ArraySeqBuilder<>(CharSeq.class);
        for (final CharSequence part : parts) {
          builder.add(CharSeq.create(part.toString()));
        }

        final int index = dict.parse(builder.build()).intAt(0);

        // skip unnecessary words
        if (index < 0) {
          return;
        }

        provider = providers[index];
      } else {
        provider = providers[providers.length - 1];
      }

      final Vec vec = converter.convertFrom(split[1]);
      provider.beta = VecTools.copySparse(vec); // optimize storage space
    });

    // TODO: is it necessary?
    double totalBigramFreq = 0;
    for(int i = 0; i < providers.length; ++i) {
      totalBigramFreq += l1(providers[i].beta);
    }

    // TODO: is it necessary?
    for(int i = 0; i < providers.length; ++i) {
      providers[i].probab = (l1(providers[i].beta) + 1) / (totalBigramFreq + providers.length);
    }

    for(int i = 0; i < providers.length; ++i) {
      providers[i].init(providers, dict);
    }
  }*/

  public void processSeq(final IntSeq prevQSeq) {
    for (int i = 0; i < prevQSeq.length(); ++i) {
      int symbol = prevQSeq.intAt(i);

      // TODO: check useless and remove
      if (freqs.size() < symbol) {
        freqs.fill(freqs.size(), symbol + 1, 0);
      }

      freqs.set(symbol, freqs.get(symbol) + 1);
      totalFreq++;
    }
  }

  public void processGeneration(final IntSeq prevQSeq,
                                final IntSeq currentQSeq,
                                final double alpha) {
    if (prevQSeq.length() * currentQSeq.length() > 10) {
      // too many variants of bipartite graph
      return;
    }

    final int variantsCount = 1 << (prevQSeq.length() * currentQSeq.length());
    final int mask = (1 << currentQSeq.length()) - 1;

    final Vec weights = new ArrayVec(variantsCount);

    for (int currVariant = 0; currVariant < variantsCount; ++currVariant) {
      double variantLogProBab = 0;

      int variant = currVariant;
      int generated = 0;

      for (int i = 0; i < prevQSeq.length(); ++i, variant >>= currentQSeq.length()) {
        final int fragment = variant & mask;
        generated |= fragment;

        // TODO: check useless and remove
        final int index = prevQSeq.intAt(i);
        if (index < 0) {
          continue;
        }

        variantLogProBab += providers[index].logP(fragment, currentQSeq);
      }

      variantLogProBab += providers[dict.size()].logP(~generated, currentQSeq);

      // Gibbs
      weights.set(currVariant, variantLogProBab);
    }

    { // Gibbs
      double sum = 0;
      double normalizer = weights.get(0);

      for (int i = 0; i < variantsCount; ++i) {
        weights.set(i, exp(weights.get(i) - normalizer));
        sum += weights.get(i);
      }

      for (int i = 0; i < GIBBS_COUNT; ++i) {
        final int bestVariant = rng.nextSimple(weights, sum);
        applyGeneration(prevQSeq, currentQSeq, alpha / GIBBS_COUNT, bestVariant);
      }
    }
  }

  private void applyGeneration(final IntSeq prevQSeq,
                               final IntSeq currentQSeq,
                               final double alpha,
                               int bestVariant) {
    final int mask = (1 << currentQSeq.length()) - 1;

    for (int i = 0; i < prevQSeq.length(); ++i, bestVariant >>= currentQSeq.length()) {
      final int fragment = bestVariant & mask;

      // TODO: check useless and remove
      final int index = prevQSeq.intAt(i);
      if (index < 0) {
        continue;
      }

      providers[index].update(fragment, currentQSeq, alpha);
    }
  }

  public void printProviders(final Writer out,
                             final boolean limit) {
    for (int i = 0; i < providers.length; ++i) {
      providers[i].print(out, limit);
    }
  }

  public void load(String inputFile) throws IOException {
    CharSeqTools.processLines(StreamTools.openTextFile(inputFile), new Action<CharSequence>() {
      final StringBuilder builder = new StringBuilder();

      public void invoke(CharSequence line) {
        if (line.equals("}")) {
          WordGenProbabilityProvider provider = new WordGenProbabilityProvider(builder.toString(), dict);
          providers[provider.providerIndex] = provider;
          builder.delete(0, builder.length());
        } else {
          builder.append(line);
        }
      }
    });
  }

  // TODO: refactor
  /*public String findTheBestExpansion(ArraySeq<CharSeq> arg) {
    final StringBuilder builder = new StringBuilder();
    final TObjectDoubleHashMap<Seq<CharSeq>> expansionScores = new TObjectDoubleHashMap<>();
    final double[] normalize = new double[1];
    dict.visitVariants(arg, freqs, totalFreq, (seq, probab) -> {
      if (probab < -100)
        return true;
      for (int i = 0; i < seq.length(); ++i) {
        if (i > 0)
          builder.append(" ");
        final int symIndex = seq.intAt(i);
        visitExpVariants(symIndex, (a, b) -> {
//          System.out.println(dict.get(a).toString() + " " + b);
          final double symProbab = b * exp(probab);
//          double logProbab = log(symProbab);
//          if (logProbab < 1e-20)
//            return false;
          normalize[0] = max(exp(probab), normalize[0]);
          expansionScores.adjustOrPutValue(dict.get(a), symProbab, symProbab);
          return true;
        }, 1.);
//        builder.append(dict.get(symIndex));
      }
//      builder.append("\t").append(probab).append("\n");
      return true;
    });
    //noinspection unchecked
    final Seq<CharSeq>[] keys = expansionScores.keys(new Seq[expansionScores.size()]);
    final double[] scores = expansionScores.values();
    final int[] order = ArrayTools.sequence(0, keys.length);
    ArrayTools.parallelSort(scores, order);
    for (int i = order.length - 1; i >= 0; i--) {
      final double prob = scores[i] / normalize[0];
      if (prob < 1e-7)
        break;

      builder.append(keys[order[i]].toString()).append(" -> ").append(prob).append("\n");
    }
    return builder.toString();
  }*/

  // TODO: refactor
  /*
  private void visitExpVariants(final int index, TIntDoubleProcedure todo, double genProb) {
    if (genProb < 1e-10 || index < 0)
      return;

    WordGenProbabilityProvider provider = providers[index];
    final Seq<CharSeq> phrase = dict.get(index);
//    System.out.println("Expanding: " + phrase);
    if (provider != null) {
      provider.visitVariants((symIndex, symProb) -> {
        final double currentGenProb = genProb * symProb;
        final WordGenProbabilityProvider symProvider = providers[symIndex];
        if (symProvider != null && symProvider.isMeaningful(index)) {
          visitExpVariants(symIndex, todo, currentGenProb);
          todo.execute(symIndex, currentGenProb);
        }
        return true;
      });
    }
  }*/
}
