package de.hlrs.starplugin.gui.ensight_export;

import de.hlrs.starplugin.gui.ensight_export.chosen_jpanels.JPanel_EnsightExport_ChosenVectors;
import de.hlrs.starplugin.gui.ensight_export.chosen_jpanels.JPanel_EnsightExport_ChosenScalars;
import de.hlrs.starplugin.gui.ensight_export.chosen_jpanels.JPanel_EnsightExport_ChosenGeometry;
import de.hlrs.starplugin.util.GetGridBagConstraints;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.JPanel;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class JPanel_EnsightExport extends JPanel {

    private final GridBagLayout layout;
    private JPanel_EnsightExport_ChosenGeometry EnsightExportChosenGeometryPanel;
    private JPanel_EnsightExport_ChosenScalars EnsightExportChosenScalarsPanel;
    private JPanel_EnsightExport_ChosenVectors EnsightExportChosenVectorsPanel;
    private JTabbedPane_EnsightExport EnsightExportTabbedPane;
    private JPanel_EnsightExport_FolderandOptions EnsightExportFolderandOptionsPanel;

    //GUI Buasteine für JPanel_EnsightExport_ChosenGeometry
    //Bausteine für JPanel Regions
    public JPanel_EnsightExport() {
        super();

        layout = new GridBagLayout();
        setLayout(layout);

        //File Destination
        EnsightExportFolderandOptionsPanel = new JPanel_EnsightExport_FolderandOptions();
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 0, 1, 0, new Insets(5, 0, 15, 0),
                GridBagConstraints.HORIZONTAL,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 6;
        this.add(EnsightExportFolderandOptionsPanel, GBCon);

        //TabbedPane (Regions,Boundaries, ScalarFuctions, Vector Functions)
        EnsightExportTabbedPane = new JTabbedPane_EnsightExport();
        GBCon = GetGridBagConstraints.get(0, 1, 1, 1, new Insets(0, 0, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 3;
        GBCon.gridwidth = 3;
        this.add(EnsightExportTabbedPane, GBCon);

        //ChosenGeometryPanel
        EnsightExportChosenGeometryPanel = new JPanel_EnsightExport_ChosenGeometry();
        GBCon = GetGridBagConstraints.get(3, 1, 1, 1, new Insets(0, 10, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 3;
        GBCon.gridwidth = 1;
        this.add(EnsightExportChosenGeometryPanel, GBCon);

        //ChosenScalarPanel
        EnsightExportChosenScalarsPanel = new JPanel_EnsightExport_ChosenScalars();
        GBCon = GetGridBagConstraints.get(4, 1, 1, 1, new Insets(0, 10, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 3;
        GBCon.gridwidth = 1;
        this.add(EnsightExportChosenScalarsPanel, GBCon);

        //ChosenVectorsPanel
        EnsightExportChosenVectorsPanel = new JPanel_EnsightExport_ChosenVectors();
        GBCon = GetGridBagConstraints.get(5, 1, 1, 1, new Insets(0, 10, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 3;
        GBCon.gridwidth = 1;
        this.add(EnsightExportChosenVectorsPanel, GBCon);

    }

    public JTabbedPane_EnsightExport getEnsightExportTabbedPane() {
        return EnsightExportTabbedPane;
    }

    public JPanel_EnsightExport_ChosenGeometry getEnsightExportChosenGeometryPanel() {
        return EnsightExportChosenGeometryPanel;
    }

    public JPanel_EnsightExport_ChosenScalars getEnsightExportChosenScalarsPanel() {
        return EnsightExportChosenScalarsPanel;
    }

    public JPanel_EnsightExport_ChosenVectors getEnsightExportChosenVectorsPanel() {
        return EnsightExportChosenVectorsPanel;
    }

    public JPanel_EnsightExport_FolderandOptions getEnsightExportFolderandOptionsPanel() {
        return EnsightExportFolderandOptionsPanel;
    }
}
