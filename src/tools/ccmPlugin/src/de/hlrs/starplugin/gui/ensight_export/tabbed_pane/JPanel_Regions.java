package de.hlrs.starplugin.gui.ensight_export.tabbed_pane;

import de.hlrs.starplugin.util.GetGridBagConstraints;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.BorderFactory;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.border.EtchedBorder;
import star.common.Region;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class JPanel_Regions extends JPanel {

    private final GridBagLayout layout;
    private JScrollPane RegionPanelScrollPane;
    private JList<Region> RegionList;

    public JPanel_Regions() {
        layout = new GridBagLayout();
        setLayout(layout);



        RegionList = new JList<Region>();
        RegionList.setVisibleRowCount(RegionList.getVisibleRowCount());
        RegionPanelScrollPane = new JScrollPane(RegionList);
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 1, 1, 1, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        RegionPanelScrollPane.setBackground(Color.white);
        RegionPanelScrollPane.setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        this.add(RegionPanelScrollPane, GBCon);

    }

    public JList getRegionList() {
        return RegionList;
    }
}
