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

    // --- Tunable Parameters (NEW - for Dynamic K) ---
    private static final double K_EARLY = 1.8; // Early in a round (large hand)
    private static final double K_MID = 1.2;   // Mid-round
    private static final double K_LATE = 0.8;  // Late in a round (small hand)
    private static final double INSURMOUNTABLE_LEAD = 20.0;
    private static final double EPSILON_GREEDY = 0.20; // 20% random


    /**
     * --- CONSTRUCTOR ---
     * Implements Requirement 3: Simple Action Pruning
     */
    public GeminiMCTSTreeNode(GeminiMCTSPlayer player, GeminiMCTSTreeNode parent, AbstractGameState gameState, Random rnd) {
        this.player = player;
        this.parent = parent;
        this.root = parent == null ? this : parent.root;
        this.state = gameState;
        this.rnd = rnd;
        this.params = player.getParameters();
        this.fmCallsCount = 0;

        this.depth = parent == null ? 0 : parent.depth + 1;

        if (gameState.isNotTerminal()) {
            // --- REQ 3: ACTION PRUNING ---
            List<AbstractAction> availableActions = player.getForwardModel().computeAvailableActions(state, player.getParameters().actionSpace);
            List<AbstractAction> prunedActions = pruneActions(availableActions, state);
            // --- END PRUNING ---

            for (AbstractAction action : prunedActions) {
                children.put(action, null);
            }
        }
    }

    /**
     * --- mctsSearch ---
     * (Restored getRoundCounter logging)
     */
    public void mctsSearch() {
        double avgTimeTaken = 0;
        double acumTimeTaken = 0;
        long remaining = 0;
        int remainingLimit = params.breakMS;
        ElapsedCpuTimer elapsedTimer = new ElapsedCpuTimer();
        if (params.budgetType == BUDGET_TIME) {
            elapsedTimer.setMaxTimeMillis(params.budget);
        }

        long startTime = System.currentTimeMillis();
        int currentRound = -1;
        if (state instanceof SGGameState) {
            // --- RESTORED ---
            currentRound = ((SGGameState) state).getRoundCounter();
        }

        int numIters = 0;
        boolean stop = false;

        while (!stop) {
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();

            GeminiMCTSTreeNode selected = treePolicy();
            double delta = selected.rollOut();
            selected.backUp(delta);

            numIters++;

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

        long elapsed = System.currentTimeMillis() - startTime;
        if (elapsed == 0) elapsed = 1;

        // --- RESTORED ---
        /* // Uncomment for performance profiling
        System.out.println(String.format(
                "Round %d: %d iterations in %dms (%.1f iter/ms)",
                currentRound, numIters, elapsed, (double)numIters/elapsed
        ));
        */
    }

    /**
     * --- treePolicy ---
     * (Standard UCT logic, as fixed previously)
     */
    private GeminiMCTSTreeNode treePolicy() {
        GeminiMCTSTreeNode cur = this;

        while (cur.state.isNotTerminal() && cur.depth < params.maxTreeDepth) {
            List<AbstractAction> unexpanded = cur.unexpandedActions();

            if (!unexpanded.isEmpty()) {
                // --- 1. EXPAND ---
                return cur.expand(unexpanded);
            } else {
                // --- 2. SELECT ---
                AbstractAction actionChosen = cur.ucb();
                if (actionChosen == null) return cur; // Terminal node
                cur = cur.children.get(actionChosen);
            }
        }
        return cur;
    }

    /**
     * --- expand ---
     * (Standard random-choice expansion)
     */
    private GeminiMCTSTreeNode expand(List<AbstractAction> notChosen) {
        Collections.shuffle(notChosen, rnd);
        AbstractAction chosen = notChosen.get(0);
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
     * Implements Requirement 4: Dynamic UCB Exploration (K)
     */
    private AbstractAction ucb() {
        AbstractAction bestAction = null;
        double bestValue = -Double.MAX_VALUE;

        // --- REQ 4: Get K based on the *root node's* current state ---
        double dynamicK = root.getDynamicK();

        // --- DEBUG LOG (FIXED) ---
        // Uncomment this line to check if K is changing
        // if (root.state instanceof SGGameState) {
        //     System.out.println("Hand: " + ((SGGameState)root.state).getPlayerHands().get(root.state.getCurrentPlayer()).getComponents().size() + ", Using dynamic K: " + dynamicK);
        // }
        // --- END DEBUG LOG ---


        for (AbstractAction action : children.keySet()) {
            GeminiMCTSTreeNode child = children.get(action);
            if (child == null) continue;

            double childValue = child.totValue / (child.nVisits + params.epsilon);

            // Use the dynamic K value for exploration
            double explorationTerm = dynamicK * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

            double uctValue = childValue + explorationTerm;
            uctValue = noise(uctValue, params.epsilon, rnd.nextDouble());

            if (uctValue > bestValue) {
                bestAction = action;
                bestValue = uctValue;
            }
        }
        if (bestAction == null) return null;
        root.fmCallsCount++;
        return bestAction;
    }

    /**
     * --- rollOut ---
     * Implements Requirement 1: Smart Rollout Policy
     * Implements Requirement 2: Early Rollout Termination
     */
    private double rollOut() {
        int rolloutDepth = 0;
        AbstractGameState rolloutState = this.state.copy();

        while (!finishRollout(rolloutState, rolloutDepth)) {

            // --- REQ 2: EARLY TERMINATION ---
            if (isInsurmountableLead(rolloutState)) {
                break;
            }

            List<AbstractAction> availableActions = player.getForwardModel().computeAvailableActions(rolloutState, player.getParameters().actionSpace);
            if (availableActions.isEmpty()) break;

            // --- REQ 1: SMART ROLLOUT (EPSILON-GREEDY) ---
            AbstractAction chosenAction = getSmartAction(rolloutState, availableActions);
            // --- END SMART ROLLOUT ---

            advance(rolloutState, chosenAction);
            rolloutDepth++;
        }

        IStateHeuristic heuristic = params.getStateHeuristic();
        double value = heuristic.evaluateState(rolloutState, player.getPlayerID());
        if (Double.isNaN(value))
            throw new AssertionError("Illegal heuristic value - should be a number");
        return value;
    }

    // --- NEW HELPER METHODS (FOR REQUIREMENTS) ---

    /**
     * --- New Helper (Req 1): getSmartAction ---
     * Implements an Epsilon-Greedy policy using a fast, lightweight heuristic.
     */
    private AbstractAction getSmartAction(AbstractGameState rolloutState, List<AbstractAction> availableActions) {
        // 20% of the time, or if not in SushiGo, pick randomly
        if (rnd.nextDouble() < EPSILON_GREEDY || !(rolloutState instanceof SGGameState)) {
            return availableActions.get(rnd.nextInt(availableActions.size()));
        }

        // 80% of the time: pick the "best" action using the fast heuristic
        SGGameState sgState = (SGGameState) rolloutState;
        int currentPlayer = sgState.getCurrentPlayer();

        AbstractAction bestAction = null;
        double bestValue = Double.NEGATIVE_INFINITY;

        for (AbstractAction action : availableActions) {
            // Use the fast, state-copy-free evaluator
            double value = evaluateActionQuick(sgState, action, currentPlayer);
            if (value > bestValue) {
                bestValue = value;
                bestAction = action;
            }
        }

        return (bestAction != null) ? bestAction : availableActions.get(rnd.nextInt(availableActions.size())); // Failsafe
    }

    /**
     * --- New Helper (Req 1): evaluateActionQuick ---
     * Lightweight "selfish" card evaluation, returns a double.
     * This is fast and does not require state copies or FM calls.
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
            case Maki:          return card.count * 1.5; // Maki is always decent
            case Pudding:       return 1.0; // Pudding is a small long-term investment
            case Tempura:       return (tempuraCount % 2 == 1) ? 5.0 : 2.5; // High value for 2nd, medium for 1st
            case Sashimi:       return (sashimiCount % 3 == 2) ? 10.0 : 3.0; // High value for 3rd, medium for 1st/2nd
            case Dumpling:      return 1.5; // Always okay
            case Wasabi:        return -0.1; // Small negative to avoid taking it raw
            case Chopsticks:    return 0.5;
            default:            return 0.5;
        }
    }

    /**
     * --- New Helper (Req 2): isInsurmountableLead ---
     * Checks if any player is winning or losing by a large margin.
     */
    private boolean isInsurmountableLead(AbstractGameState state) {
        if (!(state instanceof SGGameState)) return false;

        double minScore = Double.MAX_VALUE;
        double maxScore = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < state.getNPlayers(); i++) {
            double score = state.getGameScore(i);
            if (score < minScore) minScore = score;
            if (score > maxScore) maxScore = score;
        }

        return (maxScore - minScore > INSURMOUNTABLE_LEAD);
    }

    /**
     * --- New Helper (Req 3): pruneActions (FIXED) ---
     * Filters a list of actions to remove obviously bad moves.
     */
    private List<AbstractAction> pruneActions(List<AbstractAction> actions, AbstractGameState state) {
        if (!(state instanceof SGGameState)) return actions; // Only apply to SushiGo

        SGGameState sgState = (SGGameState) state;
        int player = sgState.getCurrentPlayer();
        List<SGCard> hand;
        try {
            // --- FIX: Called getPlayerHands() ---
            hand = sgState.getPlayerHands().get(player).getComponents();
        } catch (Exception e) {
            return actions; // Hand might not be visible (e.g., if using imperfect info)
        }

        // --- Pruning Conditions ---
        // 1. Has Nigiri in hand?
        boolean hasNigiri = hand.stream().anyMatch(c ->
                c.type == SGCard.SGCardType.SquidNigiri ||
                        c.type == SGCard.SGCardType.SalmonNigiri ||
                        c.type == SGCard.SGCardType.EggNigiri);

        // 2. Is it the last pick of the round?
        boolean isLastPick = hand.size() == 1;

        // 3. Is it the last turn of the *game*? (RESTORED)
        boolean isFinalRound = sgState.getRoundCounter() == 3; // Assuming 3 rounds
        boolean isLastTurnOfGame = isFinalRound && isLastPick;


        return actions.stream().filter(action -> {
            if (!(action instanceof ChooseCard)) return true; // Keep non-card actions

            SGCard card = (SGCard) ((ChooseCard) action).getCard(state);

            // PRUNE 1: Don't play Wasabi if no Nigiri are in hand to follow up
            if (card.type == SGCard.SGCardType.Wasabi && !hasNigiri) {
                return false;
            }

            // PRUNE 2: Don't take a single Tempura/Sashimi on the last pick of a round
            if (isLastPick && (card.type == SGCard.SGCardType.Tempura || card.type == SGCard.SGCardType.Sashimi)) {
                return false;
            }

            // PRUNE 3: Don't take Pudding on the very last turn of the game (it will score 0) (RESTORED)
            if (isLastTurnOfGame && card.type == SGCard.SGCardType.Pudding) {
                return false;
            }

            return true; // Keep the action
        }).collect(Collectors.toList());
    }

    /**
     * --- New Helper (Req 4): getDynamicK (FIXED) ---
     * Returns a different UCB exploration constant based on the *turn within the round*,
     * which is proxied by the current player's hand size.
     */
    protected double getDynamicK() {
        if (!(this.state instanceof SGGameState)) {
            return params.K; // Default
        }

        SGGameState sgState = (SGGameState) this.state;
        int player = sgState.getCurrentPlayer();
        int handSize = 0;

        try {
            // Get the size of the current player's hand
            // --- FIX: Called getPlayerHands() ---
            handSize = sgState.getPlayerHands().get(player).getComponents().size();
        } catch (Exception e) {
            return params.K; // Failsafe if hand is not available
        }

        // Adjust K based on the number of cards left in hand (turn *within* the round)
        // More cards = early in round = more exploration
        if (handSize > 6) {
            return K_EARLY; // e.g., 9, 8, 7 cards left
        } else if (handSize > 2) {
            return K_MID;   // e.g., 6, 5, 4, 3 cards left
        } else {
            return K_LATE;  // e.g., 2, 1 cards left (hands are very known)
        }
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
                if (actions.isEmpty()) return null;
                return actions.get(rnd.nextInt(actions.size()));
            }
            List<AbstractAction> childActions = new ArrayList<>(children.keySet());
            return childActions.get(rnd.nextInt(childActions.size()));
        }
        return bestAction;
    }
}