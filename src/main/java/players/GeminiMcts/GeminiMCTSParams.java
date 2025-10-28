package players.GeminiMcts;

import core.AbstractGameState;
import core.interfaces.IStateHeuristic;
import players.PlayerParameters;

import java.util.Arrays;

public class GeminiMCTSParams extends PlayerParameters {

    public double K = 1.5;
    public int rolloutLength = 8;
    public int maxTreeDepth = 100;
    public double epsilon = 1e-6;
    public IStateHeuristic heuristic = AbstractGameState::getHeuristicScore;

    public GeminiMCTSParams() {
        addTunableParameter("K", 1.5, Arrays.asList(0.0, 0.1, 1.0, Math.sqrt(2), 1.5, 3.0, 10.0));
        addTunableParameter("rolloutLength", 8, Arrays.asList(0, 3, 8, 10, 30, 100));
        addTunableParameter("maxTreeDepth", 100, Arrays.asList(1, 3, 10, 30, 100));
        addTunableParameter("epsilon", 1e-6);
        addTunableParameter("heuristic", (IStateHeuristic) AbstractGameState::getHeuristicScore);
    }

    @Override
    public void _reset() {
        super._reset();
        K = (double) getParameterValue("K");
        rolloutLength = (int) getParameterValue("rolloutLength");
        maxTreeDepth = (int) getParameterValue("maxTreeDepth");
        epsilon = (double) getParameterValue("epsilon");
        heuristic = (IStateHeuristic) getParameterValue("heuristic");
    }

    @Override
    protected GeminiMCTSParams _copy() {
        return new GeminiMCTSParams();
    }

    @Override
    public IStateHeuristic getStateHeuristic() {
        return heuristic;
    }

    @Override
    public GeminiMCTSPlayer instantiate() {
        return new GeminiMCTSPlayer((GeminiMCTSParams) this.copy());
    }
}
