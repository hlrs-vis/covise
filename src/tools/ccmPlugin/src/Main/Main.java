package Main;

import star.coremodule.actions.ActiveSimulationAction;
import star.common.Simulation;

/**
 * StarCCM+ Plugin for automated generation of visualizations with COVISE
 * Main
 * @author Weiss HLRS Stuttgart
 */
@star.locale.annotation.StarAction(display = "CovisePlugin", hint = "Export and Visualize Solution in Covise")
public class Main extends ActiveSimulationAction {

    private static final String ICON = "x24.png"; // NOI18N
    private PluginContainer PC;

    @Override
    protected String iconResource() {
        return ICON;
    }

    public void performAction() {
       
        Simulation sim = this.getSimulationProcessObject().getSimulation();
        PC = new PluginContainer(sim);

    }
}
