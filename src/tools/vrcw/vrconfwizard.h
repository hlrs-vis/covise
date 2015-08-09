#ifndef VRCONFWIZARD_H
#define VRCONFWIZARD_H

#include <QMainWindow>
#include "ui_vrconfwizard.h"

#include "vrcwbase.h"
#include "vrcwutils.h"

class VRCWStart;
class VRCWProjectionHw;
class VRCWProjectionDimPowerwall;
class VRCWProjectionDimCave;
class VRCWHost;
class VRCWHostProjection;
class VRCWTrackingHw;
class VRCWTrackingDim;
class VRCWFinal;


class VRConfWizard : public QMainWindow, public VRCWBase
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRConfWizard(QWidget* parent = 0);
   ~VRConfWizard();


   /*****
    * functions
    *****/
   //anzeigen der ProjectonConfigDimensions fuer Powerwall oder Cave
   void changeProConfDimPowerCave(const int& index, const proKind& kind,
         const bool& tiled, const typePro& typeP, const stType& stereo,
         const bool& bothEyes, const int& graka);


protected:
   /*****
    * functions
    *****/
   //eigene Version des virtuellen Events
   //fuer den "Schliessen" Button am Hauptfenster
   void closeEvent(QCloseEvent* event);


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRConfWizardClass ui;


   /*****
    * functions
    *****/
   //generic Input Processing
   int procPageInput(const int& index) const;

   //Button Action
   bool finish();
   bool abort_vrcw();
   bool exitVRConfWizard();


   /*****
    * variables
    *****/
   //Liste der erzeugten Widgets in der Reihenfolge der Erzeugung
   //und Erscheinen beim Ausfuehren
   QList<VRCWBase*> vrcwList;

   //Widgets
   VRCWStart* start;
   VRCWProjectionHw* projectionHw;
   VRCWProjectionDimPowerwall* projectionDimPowerwall;
   VRCWProjectionDimCave* projectionDimCave;
   VRCWHost* hostWidget;
   VRCWHostProjection* hostProjection;
   VRCWTrackingHw* trackingHw;
   VRCWTrackingDim* trackingDim;
   VRCWFinal* final;


private slots:
   //Button-Action Navigation
   void next() const;
   void back() const;

   //Entgegennehmen des Modification-Status von textEdit von vrcwfinal
   //und Setzen des Modification-Status der Anwendung
   void vrcwFinalConfigModified(const bool& modified);

   //Versenden des Signals expertModeChanged
   void emitExpertModeChanged(const bool& changed);


signals:
   void expertModeChanged(const bool& changed);

};

#endif // VRCONFWIZARD_H
