package de.hlrs.starplugin.util;

import java.awt.GridBagConstraints;
import java.awt.Insets;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class GetGridBagConstraints{

    public static GridBagConstraints get(int gridx, int gridy, float weightx, float weighty, Insets i, int fill, int anchor) {
        GridBagConstraints GBCon = new GridBagConstraints();
        GBCon.gridx = gridx;
        GBCon.gridy = gridy;
        GBCon.insets = i;
        GBCon.weightx = weightx;
        GBCon.weighty = weighty;
        GBCon.fill = fill;
        GBCon.anchor = anchor;
        return GBCon;
    }
}
