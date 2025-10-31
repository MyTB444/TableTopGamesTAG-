package players.GeminiMcts;

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.components.Counter;
import core.interfaces.IStateHeuristic;
import games.sushigo.SGGameState;
import players.PlayerConstants;
import utilities.ElapsedCpuTimer;

// --- IMPORTS ---
import games.sushigo.actions.ChooseCard;
import games.sushigo.cards.SGCard;
import java.util.*;
import java.util.stream.Collectors;

import static players.PlayerConstants.*;
import static utilities.Utils.noise;

public class GeminiMCTSTreeNode {

    // --- Core MCTS Fields ---
    protected GeminiMCTSPlayer player;
    protected GeminiMCTSTreeNode parent;
    protected GeminiMCTSTreeNode root;
    protected AbstractGameState state;
    protected Random rnd;
    protected GeminiMCTSParams params;
    protected Map<AbstractAction, GeminiMCTSTreeNode> children = new HashMap<>();

    // --- Statistics ---
    protected double totValue = 0;
    protected int nVisits = 0;
    protected int fmCallsCount = 0;
    protected int depth;

    // --- Gemini Logic Fields ---
    protected AbstractAction cachedOpponentPrediction = null; // Caches prediction at shallow depths
    protected int nChildrenExpanded = 0; // For Progressive Widening

    // --- Tunable Parameters (from Prompt 5) ---
    private static final double WIDTH_PARAMETER_EARLY = 12.0; // Rounds 1-5
    private static final double WIDTH_PARAMETER_MID = 6.0;    // Rounds 6-11 (tight!)
    private static final double WIDTH_PARAMETER_LATE = 15.0;  // Rounds 12-14
    private static final int MAX_PREDICTION_DEPTH = 2;         // Only predict at depth 0, 1, 2


    /**
     * --- CONSTRUCTOR ---
     * Implements Depth-Limited Prediction Caching (Prompt 3 & 4)
     */
    public GeminiMCTSTreeNode(GeminiMCTSPlayer player, GeminiMCTSTreeNode parent, AbstractGameState gameState, Random rnd) {
        this.player = player;
        this.parent = parent;
        this.root = parent == null ? this : parent.root;
        this.state = gameState;
        this.rnd = rnd;
        this.params = player.getParameters();
        this.fmCallsCount = 0;
        this.cachedOpponentPrediction = null;
        this.nChildrenExpanded = 0;

        // --- PROMPT 3: Track depth ---
        this.depth = parent == null ? 0 : parent.depth + 1;

        if (gameState.isNotTerminal()) {
            List<AbstractAction> availableActions = player.getForwardModel().computeAvailableActions(state, player.getParameters().actionSpace);
            int rootPlayer = player.getPlayerID();
            int currentPlayer = gameState.getCurrentPlayer();

            // --- PROMPT 3 & 5: CRITICAL DEPTH CHECK ---
            // Only cache predictions at shallow depths to prevent tree explosion
            if (currentPlayer != rootPlayer &&
                    this.depth <= MAX_PREDICTION_DEPTH && // KEY CHANGE
                    gameState instanceof SGGameState) {

                SGGameState sgState = (SGGameState) gameState;
                if (sgState.isHandKnown(rootPlayer, currentPlayer) && !availableActions.isEmpty()) {

                    // --- PROMPT 4: Debugging ---
                    // System.out.println("Predicting at depth " + this.depth);

                    this.cachedOpponentPrediction = this.predictOpponentAction(sgState, availableActions, currentPlayer);
                }
            }
            // --- END DEPTH CHECK ---

            for (AbstractAction action : availableActions) {
                children.put(action, null);
            }
        }
    }

    /**
     * --- mctsSearch ---
     * Implements Performance Monitoring (Prompt 4)
     */
    public void mctsSearch() {

        // --- 1. Original Budget Declarations (FIXED) ---
        double avgTimeTaken = 0;
        double acumTimeTaken = 0;
        long remaining = 0;
        int remainingLimit = params.breakMS;
        ElapsedCpuTimer elapsedTimer = new ElapsedCpuTimer();
        if (params.budgetType == BUDGET_TIME) {
            elapsedTimer.setMaxTimeMillis(params.budget);
        }

        // --- 2. New Performance Monitoring (PROMPT 4) ---
        long startTime = System.currentTimeMillis();
        int currentRound = -1;
        if (state instanceof SGGameState) {
            currentRound = ((SGGameState) state).getRoundCounter();
        }

        // --- 3. Main Loop ---
        int numIters = 0;
        boolean stop = false;

        while (!stop) {
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();

            // 1. Selection + Expansion
            GeminiMCTSTreeNode selected = treePolicy();
            // 2. Simulation (Rollout)
            double delta = selected.rollOut();
            // 3. Backpropagation
            selected.backUp(delta);

            numIters++;

            // --- 4. Stopping Condition (FIXED) ---
            PlayerConstants budgetType = params.budgetType;
            if (budgetType == BUDGET_TIME) {
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis());
                avgTimeTaken = acumTimeTaken / numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            } else if (budgetType == BUDGET_ITERATIONS) {
                stop = numIters >= params.budget;
            } else if (budgetType == BUDGET_FM_CALLS) {
                stop = fmCallsCount > params.budget;
            }
        }

        // --- 5. New Enhanced Logging (PROMPT 4) ---
        long elapsed = System.currentTimeMillis() - startTime;
        if (elapsed == 0) elapsed = 1; // Avoid divide by zero
        System.out.println(String.format(
                "Round %d: %d iterations in %dms (%.1f iter/ms)",
                currentRound, numIters, elapsed, (double)numIters/elapsed
        ));
    }

    /**
     * --- treePolicy ---
     * Implements Fallback for Deep Nodes (Prompt 3)
     */
    private GeminiMCTSTreeNode treePolicy() {
        GeminiMCTSTreeNode cur = this;

        while (cur.state.isNotTerminal() && cur.depth < params.maxTreeDepth) {

            // --- 1. CACHE-READING LOGIC (Shallow Depths) ---
            if (cur.cachedOpponentPrediction != null) {
                AbstractAction predictedAction = cur.cachedOpponentPrediction;
                if (!cur.children.containsKey(predictedAction)) {
                    // Failsafe
                    return standardMCTSStep(cur);
                }
                if (cur.children.get(predictedAction) == null) {
                    return cur.expandSpecificAction(predictedAction);
                } else {
                    cur = cur.children.get(predictedAction);
                    continue;
                }

                // --- 2. PROMPT 3: FALLBACK (Deep Known Opponent) ---
                // This is critical to prevent tree explosion at deep levels
            } else if (cur.depth > MAX_PREDICTION_DEPTH && isOpponentWithKnownHand(cur.state)) {

                // Deep node with known hand but no cache - use random sampling
                List<AbstractAction> actions = new ArrayList<>(cur.children.keySet()); // Use cached actions
                if (!actions.isEmpty()) {
                    AbstractAction randomAction = actions.get(rnd.nextInt(actions.size()));
                    if (cur.children.get(randomAction) == null) {
                        return cur.expandSpecificAction(randomAction);
                    } else {
                        cur = cur.children.get(randomAction);
                        continue;
                    }
                }
            }
            // --- END FALLBACK ---

            // --- 3. Standard MCTS with progressive widening ---
            return standardMCTSStep(cur);
        }
        return cur;
    }

    /**
     * --- standardMCTSStep ---
     * Implements Adaptive Width (Prompt 2 & 5)
     */
    private GeminiMCTSTreeNode standardMCTSStep(GeminiMCTSTreeNode cur) {
        List<AbstractAction> unexpanded = cur.unexpandedActions();

        // --- PROMPT 2 & 5: Use adaptive width ---
        int maxChildren = cur.getMaxChildren();

        if (!unexpanded.isEmpty() && cur.nChildrenExpanded < maxChildren) {
            // --- PROMPT 1: expand() now expands the best actions first ---
            return cur.expand(unexpanded);
        } else {
            AbstractAction actionChosen = cur.ucb();
            if (actionChosen == null) return cur; // All expanded children are terminal
            return cur.children.get(actionChosen);
        }
    }

    /**
     * --- expand ---
     * Implements Ordered Expansion (Prompt 1)
     */
    private GeminiMCTSTreeNode expand(List<AbstractAction> notChosen) {

        // --- PROMPT 1: Order unexpanded actions by predicted value ---
        int currentPlayer = state.getCurrentPlayer();

        if (currentPlayer != player.getPlayerID() && state instanceof SGGameState) {
            // For opponents: use evaluateActionQuick logic to order
            notChosen.sort((a, b) -> {
                double valueA = evaluateActionQuick((SGGameState)state, a, currentPlayer);
                double valueB = evaluateActionQuick((SGGameState)state, b, currentPlayer);
                return Double.compare(valueB, valueA); // Descending (best first)
            });
        } else {
            // For self: keep it random (or could use heuristic)
            Collections.shuffle(notChosen, rnd);
        }

        // Take first (best or random) unexpanded action
        AbstractAction chosen = notChosen.get(0);

        this.nChildrenExpanded++;
        return expandSpecificAction(chosen);
    }

    // (Helper: unexpandedActions - unchanged)
    private List<AbstractAction> unexpandedActions() {
        List<AbstractAction> unexpanded = new ArrayList<>();
        for (AbstractAction action : children.keySet()) {
            if (children.get(action) == null) {
                unexpanded.add(action);
            }
        }
        return unexpanded;
    }

    // (Helper: expandSpecificAction - unchanged)
    private GeminiMCTSTreeNode expandSpecificAction(AbstractAction chosen) {
        AbstractGameState nextState = state.copy();
        advance(nextState, chosen.copy());
        GeminiMCTSTreeNode tn = new GeminiMCTSTreeNode(player, this, nextState, rnd);
        children.put(chosen, tn);
        return tn;
    }

    // (Helper: advance - unchanged)
    private void advance(AbstractGameState gs, AbstractAction act) {
        player.getForwardModel().next(gs, act);
        root.fmCallsCount++;
    }

    /**
     * --- ucb ---
     * (Unchanged - still only checks non-null children)
     */
    private AbstractAction ucb() {
        AbstractAction bestAction = null;
        double bestValue = -Double.MAX_VALUE;

        for (AbstractAction action : children.keySet()) {
            GeminiMCTSTreeNode child = children.get(action);
            if (child == null) continue; // PW: Only check expanded

            double childValue = child.totValue / (child.nVisits + params.epsilon);
            double explorationTerm = params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));
            double uctValue = childValue + explorationTerm;
            uctValue = noise(uctValue, params.epsilon, rnd.nextDouble());

            if (uctValue > bestValue) {
                bestAction = action;
                bestValue = uctValue;
            }
        }
        if (bestAction == null) return null; // No expanded children
        root.fmCallsCount++;
        return bestAction;
    }

    /**
     * --- rollOut ---
     * (Unchanged - still 100% fast and random)
     */
    private double rollOut() {
        int rolloutDepth = 0;
        AbstractGameState rolloutState = this.state.copy();
        while (!finishRollout(rolloutState, rolloutDepth)) {
            List<AbstractAction> availableActions = player.getForwardModel().computeAvailableActions(rolloutState, player.getParameters().actionSpace);
            if (availableActions.isEmpty()) break;
            AbstractAction chosenAction = availableActions.get(rnd.nextInt(availableActions.size()));
            advance(rolloutState, chosenAction);
            rolloutDepth++;
        }
        IStateHeuristic heuristic = params.getStateHeuristic();
        double value = heuristic.evaluateState(rolloutState, player.getPlayerID());
        if (Double.isNaN(value))
            throw new AssertionError("Illegal heuristic value - should be a number");
        return value;
    }

    // --- NEW/MODIFIED HELPER METHODS ---

    /**
     * --- New Helper: evaluateActionQuick (Prompt 1) ---
     * Lightweight "selfish" card evaluation, returns a double.
     * This is the logic used by predictOpponentAction.
     */
    private double evaluateActionQuick(SGGameState state, AbstractAction action, int playerToEvaluate) {
        if (!(action instanceof games.sushigo.actions.ChooseCard)) return 0.0;

        SGCard card = (SGCard) ((games.sushigo.actions.ChooseCard) action).getCard(state);

        Map<SGCard.SGCardType, Counter> board = state.getPlayedCardTypes()[playerToEvaluate];
        int tempuraCount = board.get(SGCard.SGCardType.Tempura).getValue();
        int sashimiCount = board.get(SGCard.SGCardType.Sashimi).getValue();
        boolean hasWasabi = board.get(SGCard.SGCardType.Wasabi).getValue() > 0;

        switch (card.type) {
            case SquidNigiri:   return hasWasabi ? 9.0 : 3.0;
            case SalmonNigiri:  return hasWasabi ? 6.0 : 2.0;
            case EggNigiri:     return hasWasabi ? 3.0 : 1.0;
            case Maki:          return card.count * 1.5;
            case Pudding:       return 1.0;
            case Tempura:       return (tempuraCount % 2 == 1) ? 5.0 : 2.5;
            case Sashimi:       return (sashimiCount % 3 == 2) ? 10.0 : 3.0;
            case Dumpling:      return 1.5;
            case Wasabi:        return -0.1;
            case Chopsticks:    return 0.5;
            default:            return 0.5;
        }
    }

    /**
     * --- predictOpponentAction ---
     * (Now just finds the max from the quick evaluator)
     */
    private AbstractAction predictOpponentAction(SGGameState state, List<AbstractAction> actions, int opponentID) {
        AbstractAction bestAction = null;
        double bestValue = Double.NEGATIVE_INFINITY;

        for (AbstractAction action : actions) {
            double value = evaluateActionQuick(state, action, opponentID);
            if (value > bestValue) {
                bestValue = value;
                bestAction = action;
            }
        }

        if (bestAction == null) { // Failsafe if all actions have 0 value
            return actions.get(rnd.nextInt(actions.size()));
        }
        return bestAction;
    }

    /**
     * --- New Helper: getWidthParameter (Prompt 5) ---
     */
    private double getWidthParameter() {
        if (state instanceof SGGameState) {
            int round = ((SGGameState) state).getRoundCounter();
            if (round <= 5) return WIDTH_PARAMETER_EARLY;
            else if (round <= 11) return WIDTH_PARAMETER_MID;
            else return WIDTH_PARAMETER_LATE;
        }
        return 10.0; // Default
    }

    /**
     * --- New Helper: getMaxChildren (Prompt 2 & 5) ---
     */
    private int getMaxChildren() {
        double width = getWidthParameter();
        return Math.max(1, (int) (width * Math.sqrt(nVisits)));
    }

    /**
     * --- New Helper: isOpponentWithKnownHand (Prompt 3) ---
     */
    private boolean isOpponentWithKnownHand(AbstractGameState curState) {
        int rootPlayer = player.getPlayerID();
        int currentPlayer = curState.getCurrentPlayer();

        if (currentPlayer != rootPlayer && curState instanceof SGGameState) {
            return ((SGGameState) curState).isHandKnown(rootPlayer, currentPlayer);
        }
        return false;
    }

    // --- STANDARD HELPER METHODS (UNCHANGED) ---

    private boolean finishRollout(AbstractGameState rollerState, int depth) {
        if (depth >= params.rolloutLength)
            return true;
        return !rollerState.isNotTerminal();
    }

    private void backUp(double result) {
        GeminiMCTSTreeNode n = this;
        while (n != null) {
            n.nVisits++;
            n.totValue += result;
            n = n.parent;
        }
    }

    public AbstractAction bestAction() {
        double bestValue = -Double.MAX_VALUE;
        AbstractAction bestAction = null;
        for (AbstractAction action : children.keySet()) {
            if (children.get(action) != null) {
                GeminiMCTSTreeNode node = children.get(action);
                double childValue = node.nVisits;
                childValue = noise(childValue, params.epsilon, rnd.nextDouble());
                if (childValue > bestValue) {
                    bestValue = childValue;
                    bestAction = action;
                }
            }
        }
        if (bestAction == null) {
            if (children.isEmpty()) {
                List<AbstractAction> actions = player.getForwardModel().computeAvailableActions(state, player.getParameters().actionSpace);
                return actions.get(rnd.nextInt(actions.size()));
            }
            List<AbstractAction> childActions = new ArrayList<>(children.keySet());
            return childActions.get(rnd.nextInt(childActions.size()));
        }
        return bestAction;
    }
}