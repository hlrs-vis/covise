package de.hlrs.starplugin.interfaces;

import de.hlrs.starplugin.covise_net_generation.constructs.Construct;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public interface Interface_CoviseNetGeneration_DataChangedListener {

        void ConstructListChanged();

        void CoviseNetGenerationExportPathChanged();

        void SelectionChanged(Construct Con);
}
