package de.hlrs.starplugin.util;

import java.util.Comparator;
import java.util.Enumeration;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.MutableTreeNode;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class SortJTree {
    //Baum alphabetisch sortieren

    public static void sortTree(DefaultMutableTreeNode root) {
        Enumeration e = root.depthFirstEnumeration();
        while (e.hasMoreElements()) {
            DefaultMutableTreeNode node = (DefaultMutableTreeNode) e.nextElement();
            if (!node.isLeaf()) {
                sort(node);   //selection sort
            }
        }
    }
    public static Comparator<DefaultMutableTreeNode> tnc = new Comparator<DefaultMutableTreeNode>() {

        @Override
        public int compare(DefaultMutableTreeNode a, DefaultMutableTreeNode b) {
            String ab = a.getUserObject().toString();
            String ba = b.getUserObject().toString();
            int i = ab.compareTo(ba);
            return i;
        }
    };

    private static void sort(DefaultMutableTreeNode parent) {
        int n = parent.getChildCount();

        for (int i = 0; i < n - 1; i++) {
            int min = i;
            for (int j = i + 1; j < n; j++) {
                if (tnc.compare((DefaultMutableTreeNode) parent.getChildAt(min), (DefaultMutableTreeNode) parent.
                        getChildAt(j)) > 0) {
                    min = j;
                }
            }
            if (i != min) {
                MutableTreeNode a = (MutableTreeNode) parent.getChildAt(i);
                MutableTreeNode b = (MutableTreeNode) parent.getChildAt(min);
                parent.insert(b, i);
                parent.insert(a, min);
            }
        }
    }
}
