package de.hlrs.starplugin.interfaces;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public interface Interface_EnsightExport_DataChangedListener {

    void RegionSelectionChanged();

    void BoundarySelectionChanged();

    void ScalarsSelectionChanged();

    void VectorsSelectionChanged();

    void EnsightExportPathChanged();

    void ExportonVerticesChangedChanged();

    void AppendToExistingFileChanged();
}


