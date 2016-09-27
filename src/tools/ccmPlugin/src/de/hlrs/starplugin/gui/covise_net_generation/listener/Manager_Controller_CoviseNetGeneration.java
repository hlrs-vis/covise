package de.hlrs.starplugin.gui.covise_net_generation.listener;

import Main.PluginContainer;
import java.awt.event.ActionListener;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Manager_Controller_CoviseNetGeneration {

    public static void add(PluginContainer PC) {

        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanel_CreatedConstructsList().
                getJButton_DeleteVisualizationConstruct().addActionListener(new ActionListener_Button_Delete(
                PC));
        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanel_CreatedConstructsList().
                getJButton_CloneVisualizationConstruct().addActionListener(
                new ActionListener_Button_Clone(PC));

        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanel_CreatedConstructsList().
                getJButton_Test().addActionListener(new ActionListener_Button_SimVis(PC));


        ActionListener TypeButtons = new ActionListener_Button_Type(PC);
        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJButton_CreateVisalizationConsturct_Geometry().addActionListener(TypeButtons);
        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJButton_CreateVisalizationConsturct_CuttingSurface().addActionListener(TypeButtons);
        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJButton_CreateVisalizationConsturct_CuttingSurfaceSeries().addActionListener(TypeButtons);
        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJButton_CreateVisalizationConsturct_Streamline().addActionListener(TypeButtons);
        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJButton_CreateVisalizationConsturct_IsoSurface().addActionListener(TypeButtons);



    }
}
