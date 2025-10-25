package games.sushigo;

import core.AbstractGameState;
import core.AbstractParameters;
import core.components.Counter;
import core.interfaces.IStateHeuristic;
import evaluation.optimisation.TunableParameters;
import games.sushigo.cards.SGCard.SGCardType;

import java.util.Arrays;
import java.util.Map;

public class SGHeuristicGemini extends TunableParameters implements IStateHeuristic {

    // Rebalanced weights to prioritize forward-looking evaluation over reactive scoring.
    double potentialWeight = 0.35;
    double puddingStrategyWeight = 0.25;
    double setProgressAndRiskWeight = 0.20;
    double makiEfficiencyWeight = 0.10;
    double currentScoreWeight = 0.10;

    public SGHeuristicGemini() {
        addTunableParameter("potentialWeight", 0.35);
        addTunableParameter("puddingStrategyWeight", 0.25);
        addTunableParameter("setProgressAndRiskWeight", 0.20);
        addTunableParameter("makiEfficiencyWeight", 0.10);
        addTunableParameter("currentScoreWeight", 0.10);
        _reset();
    }

    @Override
    public void _reset() {
        potentialWeight = (double) getParameterValue("potentialWeight");
        puddingStrategyWeight = (double) getParameterValue("puddingStrategyWeight");
        setProgressAndRiskWeight = (double) getParameterValue("setProgressAndRiskWeight");
        makiEfficiencyWeight = (double) getParameterValue("makiEfficiencyWeight");
        currentScoreWeight = (double) getParameterValue("currentScoreWeight");
    }

    @Override
    public double evaluateState(AbstractGameState gs, int playerId) {
        SGGameState state = (SGGameState) gs;
        if (state.isNotTerminal()) {
            double totalValue = 0.0;
            totalValue += potentialWeight * evaluatePotential(state, playerId);
            totalValue += puddingStrategyWeight * evaluatePuddingStrategy(state, playerId);
            totalValue += setProgressAndRiskWeight * evaluateSetProgressAndRisk(state, playerId);
            totalValue += makiEfficiencyWeight * evaluateMakiEfficiency(state, playerId);
            totalValue += currentScoreWeight * evaluateCurrentScore(state, playerId);
            return totalValue;
        }
        return state.getPlayerResults()[playerId].value;
    }

    /**
     * CRITICAL NEW FUNCTION: Evaluates unrealized potential from setup cards (Wasabi, Chopsticks).
     */
    private double evaluatePotential(SGGameState state, int playerId) {
        double potentialValue = 0.0;
        double turnInRound = state.getTurnCounter() % state.getNPlayers();
        double maxTurnsInRound = state.getNPlayers();
        double decayFactor = 1.0 - (turnInRound / maxTurnsInRound);

        // 1. Evaluate Wasabi potential.
        int wasabiPlayedThisRound = state.getPlayedCardTypes(SGCardType.Wasabi, playerId).getValue();
        int nigiriPlayedThisRound = state.getPlayedCardTypes(SGCardType.EggNigiri, playerId).getValue() +
                state.getPlayedCardTypes(SGCardType.SalmonNigiri, playerId).getValue() +
                state.getPlayedCardTypes(SGCardType.SquidNigiri, playerId).getValue();

        int unusedWasabi = Math.max(0, wasabiPlayedThisRound - nigiriPlayedThisRound);

        if (unusedWasabi > 0) {
            // An unused Wasabi's potential is the average point gain from a future Nigiri.
            potentialValue += unusedWasabi * 5.0 * decayFactor;
        }

        // 2. Evaluate Chopsticks potential.
        if (state.getPlayedCardTypes(SGCardType.Chopsticks, playerId).getValue() > 0) {
            // Chopsticks enables powerful two-card plays. Its value is highest early.
            potentialValue += 4.0 * decayFactor;
        }

        return Math.min(1.0, potentialValue / 10.0);
    }

    /**
     * REWORKED: Implements the correct proactive Pudding strategy.
     */
    private double evaluatePuddingStrategy(SGGameState state, int playerId) {
        // FIX: Access the array of Maps correctly with [i] indexing
        Map<SGCardType, Counter>[] allGameCards = state.getPlayedCardTypesAllGame();
        int nPlayers = state.getNPlayers();
        int[] puddingCounts = new int[nPlayers];

        for (int i = 0; i < nPlayers; i++) {
            // Access the map for each player with allGameCards[i]
            Counter puddingCounter = allGameCards[i].get(SGCardType.Pudding);
            puddingCounts[i] = (puddingCounter == null) ? 0 : puddingCounter.getValue();
        }

        int myPudding = puddingCounts[playerId];
        int maxPudding = Arrays.stream(puddingCounts).max().orElse(0);
        int minPudding = Arrays.stream(puddingCounts).min().orElse(0);

        if (state.getRoundCounter() == 0) {
            return myPudding * 0.25; // Strong incentive to collect early
        } else {
            if (myPudding == maxPudding && myPudding > minPudding) {
                return 1.0; // Leading for the bonus is the best position.
            } else if (myPudding == minPudding && myPudding < maxPudding && nPlayers > 2) {
                return -1.0; // Being last and facing the penalty is the worst position.
            } else {
                return 0.1; // Middle or tied is neutral, but slightly positive to avoid last.
            }
        }
    }

    /**
     * REWORKED: Models the inherent risk associated with different sets.
     */
    private double evaluateSetProgressAndRisk(SGGameState state, int playerId) {
        SGParameters params = (SGParameters) state.getGameParameters();
        double value = 0.0;

        // Tempura (Medium Risk)
        int tempura = state.getPlayedCardTypes(SGCardType.Tempura, playerId).getValue();
        value += (tempura / 2) * params.valueTempuraPair;
        if (tempura % 2 == 1) {
            value += 1.5;
        }

        // Sashimi (High Risk)
        int sashimi = state.getPlayedCardTypes(SGCardType.Sashimi, playerId).getValue();
        value += (sashimi / 3) * params.valueSashimiTriple;
        int sashimiRemainder = sashimi % 3;
        if (sashimiRemainder == 1) {
            value -= 2.0; // PENALTY for starting a risky Sashimi set.
        } else if (sashimiRemainder == 2) {
            value += 5.0; // High potential value for being one card away.
        }

        // Dumplings (Scaling Risk)
        // FIX: valueDumpling is an array, need to access it with an index
        int dumplings = state.getPlayedCardTypes(SGCardType.Dumpling, playerId).getValue();
        if (dumplings > 0) {
            // Use array bounds checking to prevent index out of bounds
            int dumplingIndex = Math.min(dumplings - 1, params.valueDumpling.length - 1);
            int dumplingScore = params.valueDumpling[dumplingIndex];
            value += dumplingScore;
        }

        return Math.max(-1.0, Math.min(1.0, value / 25.0));
    }

    /**
     * REWORKED: Focuses on the efficiency of Maki collection, not just rank.
     */
    private double evaluateMakiEfficiency(SGGameState state, int playerId) {
        int myMaki = state.getPlayedCardTypes(SGCardType.Maki, playerId).getValue();
        if (myMaki == 0) return 0.0;

        int betterPlayers = 0;
        int tiedPlayers = 0;
        for (int i = 0; i < state.getNPlayers(); i++) {
            if (i != playerId) {
                int oppMaki = state.getPlayedCardTypes(SGCardType.Maki, i).getValue();
                if (oppMaki > myMaki) betterPlayers++;
                else if (oppMaki == myMaki) tiedPlayers++;
            }
        }

        SGParameters params = (SGParameters) state.getGameParameters();
        double expectedPoints = 0.0;
        if (betterPlayers == 0) expectedPoints = params.valueMakiMost / (1.0 + tiedPlayers);
        else if (betterPlayers == 1 && state.getNPlayers() > 2) expectedPoints = params.valueMakiSecond / (1.0 + tiedPlayers);

        // Efficiency is points per icon. Rewards getting points with fewer icons.
        double pointsPerMakiIcon = expectedPoints / myMaki;

        return Math.min(1.0, pointsPerMakiIcon / 2.0); // 2.0 PPC is a strong efficiency target
    }

    /**
     * UNCHANGED but de-prioritized.
     */
    private double evaluateCurrentScore(SGGameState state, int playerId) {
        double myScore = state.getGameScore(playerId);
        double totalDiff = 0.0;
        for (int i = 0; i < state.getNPlayers(); i++) {
            if (i != playerId) {
                totalDiff += (myScore - state.getGameScore(i));
            }
        }
        return Math.max(-1.0, Math.min(1.0, totalDiff / (20.0 * (state.getNPlayers() - 1))));
    }

    @Override
    protected AbstractParameters _copy() {
        SGHeuristicGemini copy = new SGHeuristicGemini();
        copy.potentialWeight = potentialWeight;
        copy.puddingStrategyWeight = puddingStrategyWeight;
        copy.setProgressAndRiskWeight = setProgressAndRiskWeight;
        copy.makiEfficiencyWeight = makiEfficiencyWeight;
        copy.currentScoreWeight = currentScoreWeight;
        return copy;
    }

    @Override
    protected boolean _equals(Object o) {
        if (o instanceof SGHeuristicGemini) {
            SGHeuristicGemini other = (SGHeuristicGemini) o;
            return other.potentialWeight == potentialWeight &&
                    other.puddingStrategyWeight == puddingStrategyWeight &&
                    other.setProgressAndRiskWeight == setProgressAndRiskWeight &&
                    other.makiEfficiencyWeight == makiEfficiencyWeight &&
                    other.currentScoreWeight == currentScoreWeight;
        }
        return false;
    }

    @Override
    public Object instantiate() {
        return _copy();
    }
}
