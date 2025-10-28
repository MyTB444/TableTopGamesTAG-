package players.improvedBasisMCTS;

import core.AbstractGameState;
import core.interfaces.IStateHeuristic;
import players.PlayerParameters;

import java.util.Arrays;


public class ImprovedBasicMCTSParams extends PlayerParameters {

    // Standard MCTS parameters
    public double K = 1.414;  // Lower for more focused exploitation
    public int rolloutLength = 0;  // Trust the heuristic
    public int maxTreeDepth = 12;
    public double epsilon = 1e-6;
    public IStateHeuristic heuristic = AbstractGameState::getHeuristicScore;

    // RAVE (Rapid Action Value Estimation) parameters
    public boolean useRAVE = true;
    public double raveK = 500;  // Controls how quickly RAVE influence decreases

    // Progressive widening parameters (for large action spaces)
    public boolean useProgressiveWidening = true;
    public double progressiveWideningAlpha = 0.6;  // Controls growth rate
    public double progressiveWideningC = 3.0;      // Base constant

    // Enhanced selection parameters
    public boolean useUCB1Tuned = true;
    public boolean useValueTieBreaking = true;  // Use value as tie-breaker
    public boolean useActionOrdering = true;  // Order actions by heuristic

    public ImprovedBasicMCTSParams() {
        // Standard parameters
        addTunableParameter("K", 1.414, Arrays.asList(0.0, 0.3, 0.5, 0.7, 1.0, Math.sqrt(2), 2.0));
        addTunableParameter("rolloutLength", 0, Arrays.asList(0, 3, 5, 10, 20));
        addTunableParameter("maxTreeDepth", 12, Arrays.asList(5, 10, 15, 20, 30, 50));
        addTunableParameter("epsilon", 1e-6);
        addTunableParameter("heuristic", (IStateHeuristic) AbstractGameState::getHeuristicScore);

        // RAVE parameters
        addTunableParameter("useRAVE", true);
        addTunableParameter("raveK", 500.0, Arrays.asList(100.0, 300.0, 500.0, 1000.0));

        // Progressive widening
        addTunableParameter("useProgressiveWidening", true);
        addTunableParameter("progressiveWideningAlpha", 0.6, Arrays.asList(0.25, 0.5, 0.75));
        addTunableParameter("progressiveWideningC", 3.0, Arrays.asList(1.0, 2.0, 3.0, 5.0));

        // Selection enhancements
        addTunableParameter("useUCB1Tuned", true);
        addTunableParameter("useValueTieBreaking", true);
        addTunableParameter("useActionOrdering", true);
    }

    @Override
    public void _reset() {
        super._reset();
        K = (double) getParameterValue("K");
        rolloutLength = (int) getParameterValue("rolloutLength");
        maxTreeDepth = (int) getParameterValue("maxTreeDepth");
        epsilon = (double) getParameterValue("epsilon");
        heuristic = (IStateHeuristic) getParameterValue("heuristic");

        useRAVE = (boolean) getParameterValue("useRAVE");
        raveK = (double) getParameterValue("raveK");

        useProgressiveWidening = (boolean) getParameterValue("useProgressiveWidening");
        progressiveWideningAlpha = (double) getParameterValue("progressiveWideningAlpha");
        progressiveWideningC = (double) getParameterValue("progressiveWideningC");

        useUCB1Tuned = (boolean) getParameterValue("useUCB1Tuned");
        useValueTieBreaking = (boolean) getParameterValue("useValueTieBreaking");
        useActionOrdering = (boolean) getParameterValue("useActionOrdering");
    }

    @Override
    protected ImprovedBasicMCTSParams _copy() {
        return new ImprovedBasicMCTSParams();
    }

    @Override
    public IStateHeuristic getStateHeuristic() {
        return heuristic;
    }

    @Override
    public ImprovedBasicMCTSPlayer instantiate() {
        return new ImprovedBasicMCTSPlayer((ImprovedBasicMCTSParams) this.copy());
    }
}