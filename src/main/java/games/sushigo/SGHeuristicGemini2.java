package games.sushigo;

import core.AbstractGameState;
import core.AbstractParameters;
import core.interfaces.IStateHeuristic;
import evaluation.optimisation.TunableParameters;
import games.sushigo.cards.SGCard;
import games.sushigo.cards.SGCard.SGCardType;

import java.util.HashMap;
import java.util.Map;

public class SGHeuristicGemini2 extends TunableParameters implements IStateHeuristic {

    // Multipliers for different components of the projected score.
    double potentialMultiplier = 1.0;
    double puddingMultiplier = 1.0;
    double riskMultiplier = 1.0;
    double makiMultiplier = 1.0;
    double threatMultiplier = 0.5; // Weight for considering opponent scores.

    public SGHeuristicGemini2() {
        addTunableParameter("potentialMultiplier", 1.0);
        addTunableParameter("puddingMultiplier", 1.0);
        addTunableParameter("riskMultiplier", 1.0);
        addTunableParameter("makiMultiplier", 1.0);
        addTunableParameter("threatMultiplier", 0.5);
        _reset();
    }

    @Override
    public void _reset() {
        potentialMultiplier = (double) getParameterValue("potentialMultiplier");
        puddingMultiplier = (double) getParameterValue("puddingMultiplier");
        riskMultiplier = (double) getParameterValue("riskMultiplier");
        makiMultiplier = (double) getParameterValue("makiMultiplier");
        threatMultiplier = (double) getParameterValue("threatMultiplier");
    }

    @Override
    public double evaluateState(AbstractGameState gs, int playerId) {
        SGGameState state = (SGGameState) gs;
        if (state.isNotTerminal()) {
            double myProjectedScore = calculateProjectedScore(state, playerId);
            double opponentTotalProjectedScore = 0;

            for (int i = 0; i < state.getNPlayers(); i++) {
                if (i == playerId) continue;
                opponentTotalProjectedScore += calculateProjectedScore(state, i);
            }

            double finalScore = myProjectedScore - (threatMultiplier * opponentTotalProjectedScore);

            return finalScore;
        }
        return state.getPlayerResults()[playerId].value;
    }

    private double calculateProjectedScore(SGGameState state, int playerId) {
        Map<SGCardType, Integer> playedCardCounts = new HashMap<>();
        for (SGCard card : state.getPlayedCards().get(playerId).getComponents()) {
            playedCardCounts.merge(card.type, 1, Integer::sum);
        }

        double pScore = state.getGameScore(playerId);
        double potential = potentialMultiplier * estimatePotentialPoints(state, playerId, playedCardCounts);
        double pudding = puddingMultiplier * estimatePuddingPoints(state, playerId);
        double risk = riskMultiplier * estimateSetRiskPoints(playedCardCounts);
        double maki = makiMultiplier * estimateMakiPoints(state, playerId);
        

        return pScore + potential + pudding + risk + maki;
    }

    private double estimatePotentialPoints(SGGameState state, int playerId, Map<SGCardType, Integer> playedCardCounts) {
        double potentialPoints = 0.0;
        double decayFactor = (state.getPlayerHands().get(playerId).getSize() - 1.0) / (((SGParameters)state.getGameParameters()).nCards - 1.0);

        int wasabiPlayed = playedCardCounts.getOrDefault(SGCardType.Wasabi, 0);
        int nigiriPlayed = playedCardCounts.getOrDefault(SGCardType.EggNigiri, 0) +
                playedCardCounts.getOrDefault(SGCardType.SalmonNigiri, 0) +
                playedCardCounts.getOrDefault(SGCardType.SquidNigiri, 0);
        int unusedWasabi = Math.max(0, wasabiPlayed - nigiriPlayed);
        if (unusedWasabi > 0) {
            potentialPoints += unusedWasabi * 4.0 * decayFactor;
        }

        if (playedCardCounts.getOrDefault(SGCardType.Chopsticks, 0) > 0) {
            potentialPoints += 3.0 * decayFactor;
        }

        return potentialPoints;
    }

    private double estimatePuddingPoints(SGGameState state, int playerId) {
        int nPlayers = state.getNPlayers();
        int[] puddingCounts = new int[nPlayers];
        for (int i = 0; i < nPlayers; i++) {
            puddingCounts[i] = state.getPlayedCardTypesAllGame()[i].get(SGCardType.Pudding).getValue();
        }
        int myPudding = puddingCounts[playerId];
        int maxPudding = 0;
        int minPudding = 100;
        for (int count : puddingCounts) {
            if (count > maxPudding) maxPudding = count;
            if (count < minPudding) minPudding = count;
        }
        if (maxPudding == minPudding) return 0;
        int playersWithMax = 0;
        int playersWithMin = 0;
        for (int count : puddingCounts) {
            if (count == maxPudding) playersWithMax++;
            if (count == minPudding) playersWithMin++;
        }
        double roundMultiplier = (state.getRoundCounter() + 1.0) / 3.0;
        double puddingScore = 0;
        if (myPudding == maxPudding) {
            puddingScore = 6.0 / playersWithMax;
        }
        if (nPlayers > 2 && myPudding == minPudding) {
            puddingScore = -6.0 / playersWithMin;
        }
        return puddingScore * roundMultiplier;
    }

    private double estimateSetRiskPoints(Map<SGCardType, Integer> playedCardCounts) {
        double riskValue = 0.0;
        int tempura = playedCardCounts.getOrDefault(SGCardType.Tempura, 0);
        if (tempura % 2 == 1) {
            riskValue += 1.5;
        }
        int sashimi = playedCardCounts.getOrDefault(SGCardType.Sashimi, 0);
        int sashimiRemainder = sashimi % 3;
        if (sashimiRemainder == 2) {
            riskValue += 3.0;
        } else if (sashimiRemainder == 1) {
            riskValue -= 1.0;
        }
        int dumplings = playedCardCounts.getOrDefault(SGCardType.Dumpling, 0);
        if (dumplings > 0 && dumplings < 5) {
            riskValue += dumplings * 0.5;
        }
        return riskValue;
    }

    private double estimateMakiPoints(SGGameState state, int playerId) {
        int nPlayers = state.getNPlayers();
        int[] makiCounts = new int[nPlayers];
        for (int i = 0; i < nPlayers; i++) {
            makiCounts[i] = state.getPlayedCardTypes(SGCardType.Maki, i).getValue();
        }
        int myMaki = makiCounts[playerId];
        if (myMaki == 0) return 0;
        int firstPlaceMaki = 0;
        int secondPlaceMaki = 0;
        for (int count : makiCounts) {
            if (count > firstPlaceMaki) {
                secondPlaceMaki = firstPlaceMaki;
                firstPlaceMaki = count;
            } else if (count > secondPlaceMaki && count < firstPlaceMaki) {
                secondPlaceMaki = count;
            }
        }
        int firstPlaceTies = 0;
        for (int count : makiCounts) {
            if (count == firstPlaceMaki) firstPlaceTies++;
        }
        if (myMaki == firstPlaceMaki) {
            return 6.0 / firstPlaceTies;
        }
        if (firstPlaceTies == 1 && myMaki == secondPlaceMaki && secondPlaceMaki > 0) {
            int secondPlaceTies = 0;
            for (int count : makiCounts) {
                if (count == secondPlaceMaki) secondPlaceTies++;
            }
            return 3.0 / secondPlaceTies;
        }
        return 0;
    }

    @Override
    protected AbstractParameters _copy() {
        SGHeuristicGemini2 copy = new SGHeuristicGemini2();
        copy.potentialMultiplier = potentialMultiplier;
        copy.puddingMultiplier = puddingMultiplier;
        copy.riskMultiplier = riskMultiplier;
        copy.makiMultiplier = makiMultiplier;
        copy.threatMultiplier = threatMultiplier;
        return copy;
    }

    @Override
    protected boolean _equals(Object o) {
        if (o instanceof SGHeuristicGemini2) {
            SGHeuristicGemini2 other = (SGHeuristicGemini2) o;
            return other.potentialMultiplier == potentialMultiplier &&
                    other.puddingMultiplier == puddingMultiplier &&
                    other.riskMultiplier == riskMultiplier &&
                    other.makiMultiplier == makiMultiplier &&
                    other.threatMultiplier == threatMultiplier;
        }
        return false;
    }

    @Override
    public Object instantiate() {
        return _copy();
    }
}
