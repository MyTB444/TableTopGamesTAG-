package players.GeminiMcts; // Ensure this matches your package

import core.AbstractGameState;
import core.interfaces.IStateHeuristic;
import players.PlayerParameters;

import java.util.Arrays;

public class GeminiMCTSParams extends PlayerParameters {

    public double K = 1.5; // This might become less relevant, or used as a fallback
    public int rolloutLength = 8;
    public int maxTreeDepth = 100;
    public double epsilon = 1e-6;
    public IStateHeuristic heuristic = AbstractGameState::getHeuristicScore;

    // --- NEW DYNAMIC K PARAMS ---
    public double earlyGameK = 2.0;
    public double midGameK = 0.7;
    public double lateGameK = 1.2;
    // --- END NEW PARAMS ---

    public GeminiMCTSParams() {
        // Keep existing tunable parameters
        addTunableParameter("K", 1.5); // Keep for reference or fallback
        addTunableParameter("rolloutLength", 8);
        addTunableParameter("maxTreeDepth", 100);
        addTunableParameter("epsilon", 1e-6);
        addTunableParameter("heuristic", (IStateHeuristic) AbstractGameState::getHeuristicScore);

        // --- Make new K params tunable ---
        addTunableParameter("earlyGameK", 2.0, Arrays.asList(1.0, 1.5, 2.0, 2.5));
        addTunableParameter("midGameK", 0.7, Arrays.asList(0.1, 0.5, 0.7, 1.0, 1.5));
        addTunableParameter("lateGameK", 1.2, Arrays.asList(0.7, 1.0, 1.2, 1.5, 2.0));
        // --- END ---
    }

    @Override
    public void _reset() {
        super._reset();
        K = (double) getParameterValue("K");
        rolloutLength = (int) getParameterValue("rolloutLength");
        maxTreeDepth = (int) getParameterValue("maxTreeDepth");
        epsilon = (double) getParameterValue("epsilon");
        heuristic = (IStateHeuristic) getParameterValue("heuristic");

        // --- Reset new K params ---
        earlyGameK = (double) getParameterValue("earlyGameK");
        midGameK = (double) getParameterValue("midGameK");
        lateGameK = (double) getParameterValue("lateGameK");
        // --- END ---
    }

    @Override
    protected GeminiMCTSParams _copy() {
        // TunableParameters copy should handle these new fields
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