package de.hlrs.starplugin.util;

import javax.swing.JTree;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class JTreeExpansion {
    //alle Knoten Ausklappen

    public static void expandAllNodes(JTree Baum) {
        int j = Baum.getRowCount();
        int i = 0;
        while (i < j) {
            Baum.expandRow(i);
            i += 1;
            j = Baum.getRowCount();
        }
    }
}
