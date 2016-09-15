/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.hlrs.starplugin.gui;

import de.hlrs.starplugin.interfaces.Interface_MainFrame_StatusChangedListener;
import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.ensight_export.EnsightExport_DataManager;
import de.hlrs.starplugin.util.GetGridBagConstraints;
import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.util.ArrayList;
import java.util.List;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class JFrame_MainFrame extends JFrame {
    //GUI Bausteine JFrame_MainFrame

//    private JComponent Content;
    private JPanel_MainContent JPanelMainContent;
    private JPanel_StatusBar statusbar;
    private JPanel_ButtonBar buttonbar;
    private Integer Status;
    private GridBagLayout layout;
    private JMenuItem JMenuItem_Load;
    private JMenuItem JMenuItem_Save;
    private List<Interface_MainFrame_StatusChangedListener> listeners =
            new ArrayList<Interface_MainFrame_StatusChangedListener>();

    public JFrame_MainFrame() {
        Status = Configuration_Tool.STATUS_ENSIGHTEXPORT;
        initComponents();
    }

    private void initComponents() {


        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        this.setBackground(Color.yellow);
        this.setTitle(Configuration_GUI_Strings.Titel);
        ImageIcon imageIcon = new ImageIcon(getClass().getResource("/Main/x24.png"));
        this.setIconImage(imageIcon.getImage());
//        setLayout(FrameLayout);
        GridBagConstraints GBCon;
        layout = new GridBagLayout();
//        Content = new JPanel(layout);

//        Content.setOpaque(true);
        this.setLayout(layout);
//        setContentPane(Content);


        //Statusbar Einstellung
        statusbar = new JPanel_StatusBar(this);

        //JPanel_ButtonBar Einstellung
        buttonbar = new JPanel_ButtonBar(this);

        //MainContent Einstellung
        JPanelMainContent = new JPanel_MainContent(this);

        // Load Save Menu
        JMenuBar MenuBar = new JMenuBar();
        JMenu Menu = new JMenu(Configuration_GUI_Strings.Menu_Entry1);
        JMenuItem_Load = new JMenuItem(Configuration_GUI_Strings.Load);
        JMenuItem_Save = new JMenuItem(Configuration_GUI_Strings.Save);
        Menu.add(JMenuItem_Load);
        Menu.add(JMenuItem_Save);
        MenuBar.add(Menu);
        GBCon = GetGridBagConstraints.get(0, 0, 1, 0, new Insets(1, 0, 0, 0),
                GridBagConstraints.HORIZONTAL,
                GridBagConstraints.FIRST_LINE_START);
        this.add(MenuBar, GBCon);





        //COVISE Net Generation


        //Add GUI Units
        Insets i = new Insets(0, 0, 0, 0);
        //Statusbar
        GBCon = GetGridBagConstraints.get(0, 1, 1, 0, i, GridBagConstraints.HORIZONTAL,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.insets = new Insets(5, 5, 0, 5);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(statusbar, GBCon);


        //CardLayout
        GBCon = GetGridBagConstraints.get(0, 2, 1, 1, i, GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.insets = new Insets(5, 5, 5, 5);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        GBCon.gridheight = 2;
        this.add(JPanelMainContent, GBCon);

        //Buttonbar
        GBCon = GetGridBagConstraints.get(0, 4, 1, 0, i,
                GridBagConstraints.HORIZONTAL,
                GridBagConstraints.FIRST_LINE_END);
        GBCon.insets = new Insets(5, 5, 5, 5);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(buttonbar, GBCon);

        //StatusListener einstellen
        this.addListener(buttonbar);
        this.addListener(statusbar);
        this.addListener(JPanelMainContent);
        this.StatusChanged();

        pack();
        setVisible(true);
    }

    public int getStatus() {
        return Status;
    }

    public JMenuItem getJMenuItem_Load() {
        return JMenuItem_Load;
    }

    public JMenuItem getJMenuItem_Save() {
        return JMenuItem_Save;
    }

    public void setStatus(int Status) {

        if (Configuration_Tool.STATUS_ENSIGHTEXPORT.equals(Status)) {
            this.Status = Status;
            StatusChanged();
        }
        if (Configuration_Tool.STATUS_NETGENERATION.equals(Status)) {
            this.Status = Status;
            StatusChanged();
        }
    }

    public void changeStatus(int i) {
        Status = i;
        StatusChanged();
    }

    void StatusChanged() {

        for (Interface_MainFrame_StatusChangedListener scl : listeners) {

            scl.onChange(this.Status);
        }


    }

    public void addListener(Interface_MainFrame_StatusChangedListener toAdd) {
        listeners.add(toAdd);
    }

    public JPanel_MainContent getJPanelMainContent() {
        return JPanelMainContent;
    }

    public JPanel_ButtonBar getButtonbar() {
        return buttonbar;
    }

    public JPanel_StatusBar getStatusbar() {
        return statusbar;
    }

    public void setEnsightExportDataManager(EnsightExport_DataManager EEDM) {
    }
}





