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

    // These are no longer weights, but toggles or multipliers for point estimates.
    double potentialMultiplier = 1.0;
    double puddingMultiplier = 1.0;
    double riskMultiplier = 1.0;

    public SGHeuristicGemini() {
        addTunableParameter("potentialMultiplier", 1.0);
        addTunableParameter("puddingMultiplier", 1.0);
        addTunableParameter("riskMultiplier", 1.0);
        _reset();
    }

    @Override
    public void _reset() {
        potentialMultiplier = (double) getParameterValue("potentialMultiplier");
        puddingMultiplier = (double) getParameterValue("puddingMultiplier");
        riskMultiplier = (double) getParameterValue("riskMultiplier");
    }

    @Override
    public double evaluateState(AbstractGameState gs, int playerId) {
        SGGameState state = (SGGameState) gs;
        if (state.isNotTerminal()) {
            // The new heuristic returns a "projected score" on the same scale as the game score.
            // It's the current score plus the estimated value of un-realized assets.
            double projectedScore = state.getGameScore(playerId);

            projectedScore += potentialMultiplier * estimatePotentialPoints(state, playerId);
            projectedScore += puddingMultiplier * estimatePuddingPoints(state, playerId);
            projectedScore += riskMultiplier * estimateSetRiskPoints(state, playerId);

            return projectedScore;
        }
        return state.getPlayerResults()[playerId].value;
    }

    /**
     * Estimates the point value of unrealized potential from setup cards.
     */
    private double estimatePotentialPoints(SGGameState state, int playerId) {
        double potentialPoints = 0.0;
        double turnInRound = state.getTurnCounter() % state.getNPlayers();
        double maxTurnsInRound = state.getNPlayers();
        double decayFactor = 1.0 - (turnInRound / maxTurnsInRound);

        // 1. Wasabi Potential: An unused Wasabi is worth an estimated +4 points (avg of Salmon/Squid bonus).
        int wasabiPlayed = state.getPlayedCardTypes(SGCardType.Wasabi, playerId).getValue();
        int nigiriPlayed = state.getPlayedCardTypes(SGCardType.EggNigiri, playerId).getValue() +
                state.getPlayedCardTypes(SGCardType.SalmonNigiri, playerId).getValue() +
                state.getPlayedCardTypes(SGCardType.SquidNigiri, playerId).getValue();
        int unusedWasabi = Math.max(0, wasabiPlayed - nigiriPlayed);
        if (unusedWasabi > 0) {
            potentialPoints += unusedWasabi * 4.0 * decayFactor;
        }

        // 2. Chopsticks Potential: Worth an estimated +3 points (enabling a combo better than a single pick).
        if (state.getPlayedCardTypes(SGCardType.Chopsticks, playerId).getValue() > 0) {
            potentialPoints += 3.0 * decayFactor;
        }

        return potentialPoints;
    }

    /**
     * Estimates the end-game point value of the current pudding position.
     */
    private double estimatePuddingPoints(SGGameState state, int playerId) {
        // FIX: This is an ARRAY of Maps, not a single Map
        Map<SGCardType, Counter>[] allGameCards = state.getPlayedCardTypesAllGame();
        int nPlayers = state.getNPlayers();
        int[] puddingCounts = new int[nPlayers];

        for (int i = 0; i < nPlayers; i++) {
            // FIX: Access the array element first, then get the Pudding counter
            Counter puddingCounter = allGameCards[i].get(SGCardType.Pudding);
            puddingCounts[i] = (puddingCounter == null) ? 0 : puddingCounter.getValue();
        }

        int myPudding = puddingCounts[playerId];
        int maxPudding = Arrays.stream(puddingCounts).max().orElse(0);
        int minPudding = Arrays.stream(puddingCounts).min().orElse(0);

        // Value pudding more in early rounds
        double roundMultiplier = (3.0 - state.getRoundCounter()) / 3.0;

        if (myPudding == maxPudding && myPudding > minPudding) {
            return 6.0 * roundMultiplier; // Projected +6 points
        } else if (myPudding == minPudding && myPudding < maxPudding && nPlayers > 2) {
            return -6.0 * roundMultiplier; // Projected -6 points
        }
        return 0; // Middle of the pack is neutral
    }

    /**
     * Adds a small bonus/penalty based on the risk of incomplete sets.
     * This is a minor adjustment to the main score.
     */
    private double estimateSetRiskPoints(SGGameState state, int playerId) {
        double riskValue = 0.0;

        // Tempura: Close to completing is good
        int tempura = state.getPlayedCardTypes(SGCardType.Tempura, playerId).getValue();
        if (tempura % 2 == 1) {
            riskValue += 1.5; // One away from pair
        }

        // Sashimi Risk: Being at 2 Sashimi is a high-stakes gamble.
        int sashimi = state.getPlayedCardTypes(SGCardType.Sashimi, playerId).getValue();
        int sashimiRemainder = sashimi % 3;
        if (sashimiRemainder == 2) {
            riskValue += 3.0; // High potential, give a small bonus
        } else if (sashimiRemainder == 1) {
            riskValue -= 1.0; // High risk of being a dead card, small penalty
        }

        return riskValue;
    }

    @Override
    protected AbstractParameters _copy() {
        SGHeuristicGemini copy = new SGHeuristicGemini();
        copy.potentialMultiplier = potentialMultiplier;
        copy.puddingMultiplier = puddingMultiplier;
        copy.riskMultiplier = riskMultiplier;
        return copy;
    }

    @Override
    protected boolean _equals(Object o) {
        if (o instanceof SGHeuristicGemini) {
            SGHeuristicGemini other = (SGHeuristicGemini) o;
            return other.potentialMultiplier == potentialMultiplier &&
                    other.puddingMultiplier == puddingMultiplier &&
                    other.riskMultiplier == riskMultiplier;
        }
        return false;
    }

    @Override
    public Object instantiate() {
        return _copy();
    }
}