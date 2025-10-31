package games.sushigo;

import core.AbstractGameState;
import core.AbstractParameters;
import core.components.Counter;
import core.interfaces.IStateHeuristic;
import evaluation.optimisation.TunableParameters;
import games.sushigo.SGGameState;
import games.sushigo.SGParameters;
import games.sushigo.cards.SGCard.SGCardType;

import java.util.Map;

/**
 * Improved Sushi Go Heuristic with better strategic awareness:
 * - Adaptive weights based on game phase (early/mid/late)
 * - Context-aware card valuation (e.g., Wasabi without Nigiri is worthless)
 * - Opponent modeling (blocking high-value opportunities)
 * - Marginal value analysis (diminishing returns on sets)
 * - Hand quality awareness during card selection
 */
public class SGHeuristic extends TunableParameters implements IStateHeuristic {

    // Base weights - will be adjusted dynamically
    double scoreLeadWeight = 0.25;
    double makiCompetitionWeight = 0.20;
    double setCompletionWeight = 0.20;
    double puddingStrategyWeight = 0.15;
    double cardSynergyWeight = 0.10;
    double opponentBlockingWeight = 0.10;

    public SGHeuristic() {
        addTunableParameter("scoreLeadWeight", 0.25);
        addTunableParameter("makiCompetitionWeight", 0.20);
        addTunableParameter("setCompletionWeight", 0.20);
        addTunableParameter("puddingStrategyWeight", 0.15);
        addTunableParameter("cardSynergyWeight", 0.10);
        addTunableParameter("opponentBlockingWeight", 0.10);
        _reset();
    }

    @Override
    public void _reset() {
        scoreLeadWeight = (double) getParameterValue("scoreLeadWeight");
        makiCompetitionWeight = (double) getParameterValue("makiCompetitionWeight");
        setCompletionWeight = (double) getParameterValue("setCompletionWeight");
        puddingStrategyWeight = (double) getParameterValue("puddingStrategyWeight");
        cardSynergyWeight = (double) getParameterValue("cardSynergyWeight");
        opponentBlockingWeight = (double) getParameterValue("opponentBlockingWeight");
    }

    @Override
    public double evaluateState(AbstractGameState gs, int playerId) {
        SGGameState state = (SGGameState) gs;

        // For terminal states, return the actual game result
        if (!state.isNotTerminal()) {
            return state.getPlayerResults()[playerId].value;
        }

        // Determine game phase for adaptive weighting
        GamePhase phase = determineGamePhase(state);

        double totalValue = 0.0;

        // Core evaluation components with phase-adaptive weights
        totalValue += getPhaseWeight(scoreLeadWeight, phase, 1.0, 1.2, 1.3)
                * evaluateScoreLead(state, playerId);

        totalValue += getPhaseWeight(makiCompetitionWeight, phase, 0.8, 1.2, 1.0)
                * evaluateMakiStrategy(state, playerId);

        totalValue += getPhaseWeight(setCompletionWeight, phase, 1.2, 1.0, 0.8)
                * evaluateSetCompletion(state, playerId);

        totalValue += getPhaseWeight(puddingStrategyWeight, phase, 0.7, 1.0, 1.5)
                * evaluatePuddingStrategy(state, playerId);

        totalValue += cardSynergyWeight * evaluateCardSynergies(state, playerId);

        totalValue += opponentBlockingWeight * evaluateOpponentBlocking(state, playerId, phase);

        return Math.max(-1.0, Math.min(1.0, totalValue));
    }

    /**
     * Determines the current game phase
     */
    private GamePhase determineGamePhase(SGGameState state) {
        int round = state.getRoundCounter();
        int cardsPlayed = state.getPlayedCardTypes(SGCardType.Maki, state.getCurrentPlayer()).getValue(); // rough estimate

        if (round == 0) {
            return cardsPlayed < 3 ? GamePhase.EARLY : GamePhase.MID;
        } else if (round == 1) {
            return GamePhase.MID;
        } else {
            return cardsPlayed < 4 ? GamePhase.MID : GamePhase.LATE;
        }
    }

    private double getPhaseWeight(double baseWeight, GamePhase phase,
                                  double earlyMultiplier, double midMultiplier, double lateMultiplier) {
        switch (phase) {
            case EARLY: return baseWeight * earlyMultiplier;
            case MID: return baseWeight * midMultiplier;
            case LATE: return baseWeight * lateMultiplier;
            default: return baseWeight;
        }
    }

    /**
     * Enhanced score evaluation with non-linear scaling
     */
    private double evaluateScoreLead(SGGameState state, int playerId) {
        double myScore = state.getGameScore(playerId);
        int nPlayers = state.getNPlayers();

        double[] opponentScores = new double[nPlayers - 1];
        int idx = 0;
        for (int i = 0; i < nPlayers; i++) {
            if (i != playerId) {
                opponentScores[idx++] = state.getGameScore(i);
            }
        }

        // Sort to find leader and average
        java.util.Arrays.sort(opponentScores);
        double bestOpponent = opponentScores[opponentScores.length - 1];
        double avgOpponent = java.util.Arrays.stream(opponentScores).average().orElse(0.0);

        // Being ahead of the leader is most important
        double leadValue = (myScore - bestOpponent) / 30.0; // more aggressive scaling

        // Being above average is good
        double avgValue = (myScore - avgOpponent) / 40.0;

        // Non-linear scaling: bigger leads are disproportionately better
        double combinedValue = 0.7 * leadValue + 0.3 * avgValue;
        return Math.tanh(combinedValue); // smooth S-curve
    }

    /**
     * Strategic Maki evaluation with marginal value and blocking
     */
    private double evaluateMakiStrategy(SGGameState state, int playerId) {
        SGParameters params = (SGParameters) state.getGameParameters();
        int nPlayers = state.getNPlayers();

        int myMaki = state.getPlayedCardTypes(SGCardType.Maki, playerId).getValue();

        // Build opponent Maki counts
        int[] opponentMaki = new int[nPlayers - 1];
        int idx = 0;
        for (int i = 0; i < nPlayers; i++) {
            if (i != playerId) {
                opponentMaki[idx++] = state.getPlayedCardTypes(SGCardType.Maki, i).getValue();
            }
        }
        java.util.Arrays.sort(opponentMaki);

        int firstPlace = opponentMaki[opponentMaki.length - 1];
        int secondPlace = opponentMaki.length > 1 ? opponentMaki[opponentMaki.length - 2] : 0;

        // Calculate current expected value
        double currentExpectedValue = calculateMakiExpectedValue(myMaki, firstPlace, secondPlace, nPlayers, params);

        // Calculate marginal value of one more Maki
        double marginalValue = calculateMakiExpectedValue(myMaki + 1, firstPlace, secondPlace, nPlayers, params)
                - currentExpectedValue;

        // Normalize: max Maki value is 6 points
        double normalizedCurrent = currentExpectedValue / 6.0;
        double normalizedMarginal = marginalValue / 6.0;

        // Weight current position and marginal value
        return 0.6 * normalizedCurrent + 0.4 * normalizedMarginal;
    }

    private double calculateMakiExpectedValue(int myMaki, int firstPlace, int secondPlace,
                                              int nPlayers, SGParameters params) {
        if (myMaki == 0) return 0.0;

        int betterCount = 0;
        if (firstPlace > myMaki) betterCount++;
        if (secondPlace > myMaki && secondPlace != firstPlace) betterCount++;

        // Exact ties assumed split evenly (simplified)
        if (myMaki > firstPlace) {
            return params.valueMakiMost; // Clear first
        } else if (myMaki == firstPlace) {
            return params.valueMakiMost / 2.0; // Tied for first
        } else if (betterCount == 1 && nPlayers > 2) {
            if (myMaki > secondPlace) {
                return params.valueMakiSecond; // Clear second
            } else if (myMaki == secondPlace) {
                return params.valueMakiSecond / 2.0; // Tied for second
            }
        }

        return 0.0;
    }

    /**
     * Set completion with diminishing returns and opportunity cost
     */
    private double evaluateSetCompletion(SGGameState state, int playerId) {
        SGParameters params = (SGParameters) state.getGameParameters();
        double value = 0.0;

        // Tempura - pairs worth 5 points
        int tempura = state.getPlayedCardTypes(SGCardType.Tempura, playerId).getValue();
        int tempuraPairs = tempura / 2;
        int tempuraRemainder = tempura % 2;

        // Completed sets: full value
        value += (tempuraPairs * params.valueTempuraPair) / 10.0;

        // Incomplete sets: higher value if close to completion
        if (tempuraRemainder == 1) {
            // One card from completing - high marginal value
            value += 0.4; // worth pursuing
        }

        // Sashimi - triples worth 10 points (harder to complete)
        int sashimi = state.getPlayedCardTypes(SGCardType.Sashimi, playerId).getValue();
        int sashimiTriples = sashimi / 3;
        int sashimiRemainder = sashimi % 3;

        value += (sashimiTriples * params.valueSashimiTriple) / 10.0;

        if (sashimiRemainder == 2) {
            value += 0.6; // Very close - high priority
        } else if (sashimiRemainder == 1) {
            value += 0.15; // Far away - low value unless early game
        }

        // Dumplings - diminishing returns
        int dumplings = state.getPlayedCardTypes(SGCardType.Dumpling, playerId).getValue();
        if (dumplings > 0 && dumplings <= params.valueDumpling.length) {
            int currentValue = params.valueDumpling[Math.min(dumplings - 1, params.valueDumpling.length - 1)];
            value += currentValue / 15.0;

            // Calculate marginal value (should we get more?)
            if (dumplings < params.valueDumpling.length - 1) {
                int nextValue = params.valueDumpling[dumplings];
                int marginalGain = nextValue - currentValue;

                // After 3-4 dumplings, marginal value drops significantly
                if (marginalGain <= 1) {
                    value -= 0.1; // penalty for over-investing
                }
            }
        }

        return Math.min(1.2, value);
    }

    /**
     * Strategic pudding evaluation based on game phase and position
     */
    private double evaluatePuddingStrategy(SGGameState state, int playerId) {
        Map<SGCardType, Counter>[] allGameCards = state.getPlayedCardTypesAllGame();
        int myPudding = allGameCards[playerId].get(SGCardType.Pudding).getValue();

        int nPlayers = state.getNPlayers();
        int[] opponentPuddings = new int[nPlayers - 1];
        int idx = 0;
        for (int i = 0; i < nPlayers; i++) {
            if (i != playerId) {
                opponentPuddings[idx++] = allGameCards[i].get(SGCardType.Pudding).getValue();
            }
        }
        java.util.Arrays.sort(opponentPuddings);

        int maxPudding = opponentPuddings[opponentPuddings.length - 1];
        int minPudding = opponentPuddings[0];

        SGParameters params = (SGParameters) state.getGameParameters();
        int round = state.getRoundCounter();
        double roundWeight = (round + 1) / 3.0; // importance scales with rounds

        // Strategic thresholds
        boolean inFirstPlace = myPudding >= maxPudding;
        boolean inLastPlace = myPudding <= minPudding;
        boolean saflyAheadOfLast = myPudding >= minPudding + 2;

        double value = 0.0;

        if (inFirstPlace) {
            // Winning pudding is good, but diminishing returns past +2 lead
            int lead = myPudding - maxPudding;
            double baseValue = params.valuePuddingMost / 6.0;

            if (lead > 2) {
                baseValue *= 0.8; // reduced value for excessive lead
            }
            value = roundWeight * baseValue;

        } else if (inLastPlace && nPlayers > 2) {
            // Being last is BAD - negative value
            int deficit = minPudding - myPudding;
            double penalty = params.valuePuddingLeast / 6.0;

            if (deficit > 1) {
                penalty *= 1.3; // worse if significantly behind
            }
            value = roundWeight * penalty;

        } else if (saflyAheadOfLast) {
            // Safe middle position - slight positive
            value = roundWeight * 0.15;

        } else {
            // Dangerous middle (close to last)
            int distanceFromLast = myPudding - minPudding;
            value = roundWeight * (0.1 * distanceFromLast - 0.1);
        }

        return value;
    }

    /**
     * Evaluates card synergies (especially Wasabi + Nigiri)
     */
    private double evaluateCardSynergies(SGGameState state, int playerId) {
        SGParameters params = (SGParameters) state.getGameParameters();
        double value = 0.0;

        // Wasabi analysis
        int wasabi = state.getPlayedCardTypes(SGCardType.Wasabi, playerId).getValue();
        int egg = state.getPlayedCardTypes(SGCardType.EggNigiri, playerId).getValue();
        int salmon = state.getPlayedCardTypes(SGCardType.SalmonNigiri, playerId).getValue();
        int squid = state.getPlayedCardTypes(SGCardType.SquidNigiri, playerId).getValue();
        int totalNigiri = egg + salmon + squid;

        // Wasabi multiplies next Nigiri by 3 (base value already counted)
        // So Wasabi adds 2x the Nigiri value
        // Unused Wasabi is worthless (or negative value)

        if (wasabi > totalNigiri) {
            // We have unused Wasabi - bad!
            int unusedWasabi = wasabi - totalNigiri;
            value -= 0.3 * unusedWasabi; // penalty for dead cards
        } else if (wasabi > 0) {
            // We have used Wasabi - good!
            // Assume best Nigiri were played on Wasabi
            int usedWasabi = wasabi;

            // Estimate bonus value (2x best Nigiri)
            double bonusValue = usedWasabi * 2 * params.valueSquidNigiri; // assume squid
            value += bonusValue / 20.0;
        }

        // Chopsticks analysis (allows picking 2 cards)
        int chopsticks = state.getPlayedCardTypes(SGCardType.Chopsticks, playerId).getValue();
        if (chopsticks > 0) {
            // Chopsticks have strategic value but no points
            // Value based on game phase
            value += 0.2 * Math.min(chopsticks, 2); // diminishing returns
        }

        return value;
    }

    /**
     * Evaluates blocking opponents from valuable cards/positions
     */
    private double evaluateOpponentBlocking(SGGameState state, int playerId, GamePhase phase) {
        int nPlayers = state.getNPlayers();
        double blockingValue = 0.0;

        // Check if opponents are close to completing high-value sets
        for (int i = 0; i < nPlayers; i++) {
            if (i == playerId) continue;

            // Check Sashimi (10 points if completed)
            int oppSashimi = state.getPlayedCardTypes(SGCardType.Sashimi, i).getValue();
            if (oppSashimi == 2) {
                // Opponent needs 1 more Sashimi - worth blocking
                blockingValue += 0.3;
            }

            // Check Tempura (5 points if completed)
            int oppTempura = state.getPlayedCardTypes(SGCardType.Tempura, i).getValue();
            if (oppTempura % 2 == 1) {
                // Opponent needs 1 more Tempura
                blockingValue += 0.15;
            }
        }

        // In late game, blocking becomes more important
        if (phase == GamePhase.LATE) {
            blockingValue *= 1.5;
        }

        return blockingValue;
    }

    @Override
    protected AbstractParameters _copy() {
        SGHeuristic copy = new SGHeuristic();
        copy.scoreLeadWeight = scoreLeadWeight;
        copy.makiCompetitionWeight = makiCompetitionWeight;
        copy.setCompletionWeight = setCompletionWeight;
        copy.puddingStrategyWeight = puddingStrategyWeight;
        copy.cardSynergyWeight = cardSynergyWeight;
        copy.opponentBlockingWeight = opponentBlockingWeight;
        return copy;
    }

    @Override
    protected boolean _equals(Object o) {
        if (o instanceof SGHeuristic) {
            SGHeuristic other = (SGHeuristic) o;
            return other.scoreLeadWeight == scoreLeadWeight
                    && other.makiCompetitionWeight == makiCompetitionWeight
                    && other.setCompletionWeight == setCompletionWeight
                    && other.puddingStrategyWeight == puddingStrategyWeight
                    && other.cardSynergyWeight == cardSynergyWeight
                    && other.opponentBlockingWeight == opponentBlockingWeight;
        }
        return false;
    }

    @Override
    public Object instantiate() {
        return _copy();
    }

    private enum GamePhase {
        EARLY, MID, LATE
    }
}