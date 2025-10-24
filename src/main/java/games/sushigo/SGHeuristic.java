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

public class SGHeuristic extends TunableParameters implements IStateHeuristic {

    // Weights that sum to 1.0 for normalized evaluation
    double currentScoreWeight = 0.30;
    double makiCompetitionWeight = 0.25;
    double setProgressWeight = 0.20;
    double puddingPositionWeight = 0.20;
    double nigiriValueWeight = 0.05;

    public SGHeuristic() {
        addTunableParameter("currentScoreWeight", 0.30);
        addTunableParameter("makiCompetitionWeight", 0.25);
        addTunableParameter("setProgressWeight", 0.20);
        addTunableParameter("puddingPositionWeight", 0.20);
        addTunableParameter("nigiriValueWeight", 0.05);
        _reset();
    }

    @Override
    public void _reset() {
        currentScoreWeight = (double) getParameterValue("currentScoreWeight");
        makiCompetitionWeight = (double) getParameterValue("makiCompetitionWeight");
        setProgressWeight = (double) getParameterValue("setProgressWeight");
        puddingPositionWeight = (double) getParameterValue("puddingPositionWeight");
        nigiriValueWeight = (double) getParameterValue("nigiriValueWeight");
    }

    @Override
    public double evaluateState(AbstractGameState gs, int playerId) {
        SGGameState state = (SGGameState) gs;

        // For terminal states, return the actual game result
        if (!state.isNotTerminal()) {
            return state.getPlayerResults()[playerId].value;
        }

        double totalValue = 0.0;

        // Evaluate each component of the board state
        totalValue += currentScoreWeight * evaluateCurrentScore(state, playerId);
        totalValue += makiCompetitionWeight * evaluateMakiRace(state, playerId);
        totalValue += setProgressWeight * evaluateSetProgress(state, playerId);
        totalValue += puddingPositionWeight * evaluatePuddingRace(state, playerId);
        totalValue += nigiriValueWeight * evaluateNigiriCards(state, playerId);

        return totalValue;
    }

    /**
     * Evaluates current score differential with all opponents
     */
    private double evaluateCurrentScore(SGGameState state, int playerId) {
        double myScore = state.getGameScore(playerId);
        double totalDiff = 0.0;

        for (int i = 0; i < state.getNPlayers(); i++) {
            if (i != playerId) {
                totalDiff += (myScore - state.getGameScore(i));
            }
        }

        // Normalize by expected score range (roughly 150 max across 3 rounds)
        return Math.max(-1.0, Math.min(1.0, totalDiff / (50.0 * state.getNPlayers())));
    }

    /**
     * Evaluates position in Maki competition for current round
     */
    private double evaluateMakiRace(SGGameState state, int playerId) {
        Counter myMakiCounter = state.getPlayedCardTypes(SGCardType.Maki, playerId);
        int myMaki = myMakiCounter.getValue();

        if (myMaki == 0) return 0.0;

        int betterPlayers = 0;
        int tiedPlayers = 0;

        for (int i = 0; i < state.getNPlayers(); i++) {
            if (i != playerId) {
                int oppMaki = state.getPlayedCardTypes(SGCardType.Maki, i).getValue();
                if (oppMaki > myMaki) {
                    betterPlayers++;
                } else if (oppMaki == myMaki) {
                    tiedPlayers++;
                }
            }
        }

        SGParameters params = (SGParameters) state.getGameParameters();

        // Calculate expected points
        double expectedPoints = 0.0;
        if (betterPlayers == 0) {
            // First place (possibly tied)
            expectedPoints = params.valueMakiMost / (1.0 + tiedPlayers);
        } else if (betterPlayers == 1 && state.getNPlayers() > 2) {
            // Second place (possibly tied)
            expectedPoints = params.valueMakiSecond / (1.0 + tiedPlayers);
        }

        // Normalize to 0-1 range
        return expectedPoints / params.valueMakiMost;
    }

    /**
     * Evaluates progress toward completing sets (Tempura, Sashimi, Dumplings)
     */
    private double evaluateSetProgress(SGGameState state, int playerId) {
        SGParameters params = (SGParameters) state.getGameParameters();
        double value = 0.0;

        // Tempura - pairs score 5 points
        int tempura = state.getPlayedCardTypes(SGCardType.Tempura, playerId).getValue();
        int tempuraPairs = tempura / 2;
        int tempuraRemainder = tempura % 2;
        value += (tempuraPairs * params.valueTempuraPair) / 10.0;
        value += (tempuraRemainder * 0.3); // Partial value for incomplete pair

        // Sashimi - triples score 10 points
        int sashimi = state.getPlayedCardTypes(SGCardType.Sashimi, playerId).getValue();
        int sashimiTriples = sashimi / 3;
        int sashimiRemainder = sashimi % 3;
        value += (sashimiTriples * params.valueSashimiTriple) / 10.0;
        if (sashimiRemainder == 2) {
            value += 0.5; // Close to completion
        } else if (sashimiRemainder == 1) {
            value += 0.2; // Far from completion
        }

        // Dumplings - progressive scoring
        int dumplings = state.getPlayedCardTypes(SGCardType.Dumpling, playerId).getValue();
        if (dumplings > 0 && dumplings <= params.valueDumpling.length) {
            value += params.valueDumpling[Math.min(dumplings - 1, params.valueDumpling.length - 1)] / 15.0;
        }

        return Math.min(1.0, value);
    }

    /**
     * Evaluates pudding position for end-game scoring
     */
    private double evaluatePuddingRace(SGGameState state, int playerId) {
        // Get the cumulative pudding count from the array of maps
        Map<SGCardType, Counter>[] allGameCards = state.getPlayedCardTypesAllGame();
        int myPudding = allGameCards[playerId].get(SGCardType.Pudding).getValue();

        int maxPudding = myPudding;
        int minPudding = myPudding;
        int betterPlayers = 0;
        int worsePlayers = 0;

        for (int i = 0; i < state.getNPlayers(); i++) {
            if (i != playerId) {
                int oppPudding = allGameCards[i].get(SGCardType.Pudding).getValue();
                maxPudding = Math.max(maxPudding, oppPudding);
                minPudding = Math.min(minPudding, oppPudding);

                if (oppPudding > myPudding) betterPlayers++;
                if (oppPudding < myPudding) worsePlayers++;
            }
        }

        SGParameters params = (SGParameters) state.getGameParameters();
        double roundMultiplier = (state.getRoundCounter() + 1) / 3.0;

        if (myPudding == maxPudding && myPudding > 0) {
            // Leading for bonus points
            return roundMultiplier * (params.valuePuddingMost / 6.0);
        } else if (myPudding == minPudding && state.getNPlayers() > 2) {
            // Last place penalty (only in 3+ player games)
            return roundMultiplier * (params.valuePuddingLeast / 6.0);
        } else {
            // Middle position - slightly positive to avoid last
            double position = (double) worsePlayers / (state.getNPlayers() - 1);
            return roundMultiplier * (position - 0.3);
        }
    }

    /**
     * Evaluates raw Nigiri value (including Wasabi multipliers if already applied)
     */
    private double evaluateNigiriCards(SGGameState state, int playerId) {
        SGParameters params = (SGParameters) state.getGameParameters();

        int egg = state.getPlayedCardTypes(SGCardType.EggNigiri, playerId).getValue();
        int salmon = state.getPlayedCardTypes(SGCardType.SalmonNigiri, playerId).getValue();
        int squid = state.getPlayedCardTypes(SGCardType.SquidNigiri, playerId).getValue();

        // Note: If Wasabi multiplier was applied, it's already reflected in the game score,
        // so we just count base values here
        double nigiriPoints = (egg * params.valueEggNigiri +
                salmon * params.valueSalmonNigiri +
                squid * params.valueSquidNigiri);

        // Normalize (max reasonable nigiri value in a round is about 15-20)
        return nigiriPoints / 20.0;
    }

    @Override
    protected AbstractParameters _copy() {
        SGHeuristic copy = new SGHeuristic();
        copy.currentScoreWeight = currentScoreWeight;
        copy.makiCompetitionWeight = makiCompetitionWeight;
        copy.setProgressWeight = setProgressWeight;
        copy.puddingPositionWeight = puddingPositionWeight;
        copy.nigiriValueWeight = nigiriValueWeight;
        return copy;
    }

    @Override
    protected boolean _equals(Object o) {
        if (o instanceof SGHeuristic) {
            SGHeuristic other = (SGHeuristic) o;
            return other.currentScoreWeight == currentScoreWeight
                    && other.makiCompetitionWeight == makiCompetitionWeight
                    && other.setProgressWeight == setProgressWeight
                    && other.puddingPositionWeight == puddingPositionWeight
                    && other.nigiriValueWeight == nigiriValueWeight;
        }
        return false;
    }

    @Override
    public Object instantiate() {
        return _copy();
    }
}