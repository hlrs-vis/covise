#ifndef VRCWPROJECTIONDIMPOWERWALL_H
#define VRCWPROJECTIONDIMPOWERWALL_H

#include <QWidget>
#include "ui_vrcwprojectiondimpowerwall.h"

#include "vrcwbase.h"
#include "vrcwutils.h"

class VRCWProjectionResSizeTiled;
class VRCWProjectionResSize;
class VRCWProjectionVposFloor;
class wallVal;


class VRCWProjectionDimPowerwall : public QWidget, public VRCWBase
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWProjectionDimPowerwall(QWidget *parent = 0);
   ~VRCWProjectionDimPowerwall();


   /*****
    * functions
    *****/
   //Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
   //Bearbeitung und Ueberpruefung der Eingaben im GUI
   int processGuiInput(const int& index, const QList<VRCWBase*>& vrcwList);

   //Entgegennehmen der Parameter von ProjectionHw fuer die Bestimmung
   //der Projection
   void setProjectionHwParams(const proKind& kind, const bool& tiled,
         const typePro& typeP, const stType& stereo, const bool& bothEyes,
         const int& graka);


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWProjectionDimPowerwallClass ui;


   /*****
    * functions
    *****/
   //Auslesen des GUI
   QVector<int> getGuiRes() const;
   QVector<int> getGuiScreenSize() const;
   QList<wallVal*> getGuiPWallDim() const;

   //Erzeugen der Liste der Projektionen
   QStringList createProjection();


   /*****
    * variables
    *****/
   VRCWProjectionResSizeTiled* projectionResSizeTiled;
   VRCWProjectionResSize* projectionResSize;
   VRCWProjectionVposFloor* projectionVposFloor;
   QList<wallVal*> pWallDim;
   //store old values
   QString typePOld;
   QString kindOld;

   //Values from ProjectionHw
   proKind kindData;
   bool tiledData;
   typePro typePData;
   stType stereoData;
   bool bothEyesData;
   int grakaData;


   /*****
    * container
    *****/
   enum pMode {Pfpf, Pfpt, Pfat, Ptpf, Ptpt, Ptat, TVfat};

};

#endif // VRCWPROJECTIONDIMPOWERWALL_H
