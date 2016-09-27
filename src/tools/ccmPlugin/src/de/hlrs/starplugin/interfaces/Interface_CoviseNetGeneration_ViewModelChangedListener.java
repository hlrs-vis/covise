package de.hlrs.starplugin.interfaces;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public interface Interface_CoviseNetGeneration_ViewModelChangedListener {
        void CoviseNetGeneration_ExportPathChanged();

        void GeometryCard_GeometrySelectionChanged();

        void CuttingSurfaceCard_GeometrySelectionChanged();

        void CuttingSurfaceSeriesCard_GeometrySelectionChanged();

        void StreamlineCard_GeometrySelectionChanged();

        void IsoSurfaceCard_GeometrySelectionChanged();
}
