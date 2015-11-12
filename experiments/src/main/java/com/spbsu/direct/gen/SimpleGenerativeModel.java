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
import com.spbsu.direct.BroadMatch;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.io.IOException;
import java.io.Writer;

import static com.spbsu.commons.math.vectors.VecTools.l1;
import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 12.11.15
 * Time: 11:33
 */
public class SimpleGenerativeModel {
  public static final String EMPTY_ID = "##EMPTY##";


  private final WordGenProbabilityProvider[] providers;
  private final Dictionary<CharSeq> dict;
  private final FastRandom rng = new FastRandom(0);

  public SimpleGenerativeModel(Dictionary<CharSeq> dict) {
    this.dict = dict;
    this.providers = new WordGenProbabilityProvider[dict.size() + 1];
  }

  public void loadStatistics(String fileName) throws IOException {
    for (int i = 0; i < providers.length; i++) {
      providers[i] = new WordGenProbabilityProvider(dict.size(), i);
    }
    final Vec2CharSequenceConverter converter = new Vec2CharSequenceConverter();
    CharSeqTools.processLines(StreamTools.openTextFile(fileName), (Action<CharSequence>) sequence -> {
      final CharSequence[] split = CharSeqTools.split(sequence, '\t');

      final WordGenProbabilityProvider provider;
      if (!split[0].equals(EMPTY_ID)) {
        final CharSequence[] parts = CharSeqTools.split(split[0].subSequence(1, split[0].length() - 1), ", ");
        final SeqBuilder<CharSeq> builder = new ArraySeqBuilder<>(CharSeq.class);
        for (final CharSequence part : parts) {
          builder.add(new CharSeqAdapter(part.toString()));
        }

        final int index1 = dict.parse(builder.build()).intAt(0);
        if (index1 < 0)
          return;
        provider = providers[index1];
      }
      else provider = providers[providers.length - 1];
      final Vec vec = converter.convertFrom(split[1]);
      provider.beta = VecTools.copySparse(vec); // optimize storage space
    });
    double totalBigramFreq = 0;
    for(int i = 0; i < providers.length; i++) {
      totalBigramFreq += l1(providers[i].beta);
    }

    for(int i = 0; i < providers.length; i++) {
      providers[i].probab = (l1(providers[i].beta) + 1) / (totalBigramFreq + providers.length);
    }
    for(int i = 0; i < providers.length; i++) {
      providers[i].init(providers, dict);
    }
  }

  private int index = 0;
  private final TDoubleArrayList window = new TDoubleArrayList(1000);
  private double windowSum = 0;

  public double totalFreq = 0;
  public final TIntList freqs = new TIntArrayList();

  public void processSeq(IntSeq prevQSeq) {
    for (int i = 0; i < prevQSeq.length(); i++) {
      int symbol = prevQSeq.intAt(i);
      if (freqs.size() < symbol)
        freqs.fill(freqs.size(), symbol + 1, 0);
      freqs.set(symbol, freqs.get(symbol) + 1);
      totalFreq++;
    }
  }

  public void processGeneration(IntSeq prevQSeq, IntSeq currentQSeq, double alpha) {
    if (prevQSeq.length() * currentQSeq.length() > 10) // too many variants of bipartite graph
      return;
    final int variantsCount = 1 << (prevQSeq.length() * currentQSeq.length());
    final int mask = (1 << currentQSeq.length()) - 1;
    int bestVariant;
    double bestLogProBab;
    { // expectation
      final Vec weights = new ArrayVec(variantsCount);
      for (int p = 0; p < variantsCount; p++) {
        double variantLogProBab = 0;
        {
          int variant = p;
          int generated = 0;
          for (int i = 0; i < prevQSeq.length(); i++, variant >>= currentQSeq.length()) {
            final int fragment = variant & mask;
            generated |= fragment;
            final int index = prevQSeq.intAt(i);
            if (index < 0)
              continue;
            variantLogProBab += providers[index].logP(fragment, currentQSeq);
          }
          for (int i = 0; i < currentQSeq.length(); i++, generated >>= 1) {
            if ((generated & 1) == 1)
              continue;
            variantLogProBab += log(freqs.get(currentQSeq.intAt(i)) + 1.) - log(totalFreq + freqs.size());
          }
        }
        // Gibbs
        weights.set(p, variantLogProBab);
//        { // EM
//                  if (variantLogProBab > bestLogProBab) {
//                    bestLogProBab = variantLogProBab;
//                    bestVariant = p;
//                  }
//        }
      }

      { // Gibbs
        double sum = 0;
        double normalizer = weights.get(0);
        for (int i = 0; i < variantsCount; i++) {
          weights.set(i, exp(weights.get(i) - normalizer));
          sum += weights.get(i);
        }
        bestVariant = rng.nextSimple(weights, sum);
        bestLogProBab = log(weights.get(bestVariant)) + normalizer;
      }

    }
    { // maximization gradient descent step

      int generated = 0;
      windowSum += bestLogProBab;
      window.add(bestLogProBab);
      final double remove;
      if (window.size() > 100000) {
        remove = window.removeAt(0);
        windowSum -= remove;
      }

      boolean debug = BroadMatch.debug && (index % 100000 == 0);
      if (debug)
        System.out.print(windowSum / window.size() + "\t" + "\n");
//              debug = false;
      if (debug)
        System.out.println(prevQSeq + " -> " + currentQSeq + " " + bestLogProBab);
      double newProb = 0;
      for (int i = 0; i < prevQSeq.length(); i++, bestVariant >>= currentQSeq.length()) {
        final int fragment = bestVariant & mask;
        generated |= fragment;
        final int windex = prevQSeq.intAt(i);
        if (windex < 0)
          continue;
        providers[windex].update(fragment, currentQSeq, alpha, dict, debug);
        newProb += providers[windex].logP(fragment, currentQSeq);
      }
      if (debug)
        System.out.print(EMPTY_ID + " ->");
      for (int i = 0; i < currentQSeq.length(); i++, generated >>= 1) {
        if ((generated & 1) == 1)
          continue;
        final int windex = currentQSeq.intAt(i);
        if (debug)
          System.out.print(dict.get(windex));
        newProb += log(freqs.get(windex) + 1.) - log(totalFreq + freqs.size());
      }
      if (debug)
        System.out.println("\nNew probability: " + newProb);
    }
    index++;
  }

  public void print(Writer out, boolean limit) {
    for (int i = 0; i < providers.length; i++) {
      final WordGenProbabilityProvider provider = providers[i];
      provider.print(dict, out, limit);
    }
  }
}
