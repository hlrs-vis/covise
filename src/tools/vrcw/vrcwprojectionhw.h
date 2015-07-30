#ifndef VRCWPROJECTIONHW_H
#define VRCWPROJECTIONHW_H

#include <QWidget>
#include "ui_vrcwprojectionhw.h"

#include "vrcwbase.h"
#include "vrcwutils.h"
#include "vrcwprojectionressize.h"


class VRCWProjectionHw : public QWidget, public VRCWBase
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWProjectionHw(QWidget* parent = 0);
   ~VRCWProjectionHw();


   /*****
    * functions
    *****/
   //Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
   //Bearbeitung und Ueberpruefung der Eingaben im GUI
   int processGuiInput(const int& index, const QList<VRCWBase*>& vrcwList);

   //Auslesen des GUI
   QList<cWall> getGuiWalls() const;


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWProjectionHwClass ui;


   /*****
    * functions
    *****/
   //Auslesen des GUI
   proKind getGuiKind() const;
   bool getGuiTiled() const;
   typePro getTypePro() const;
   stType getGuiStereo() const;
   bool getGuiBothEyes() const;
   int getGuiGraka() const;
   bool getCtrlMon() const;
   loRoOrient getCMon3DOrient() const;

   /*****
    * variables
    *****/
   VRCWProjectionResSize* ctrlMonRes;
   bool expertMode;


private slots:
   //die Optionen fuer kind anzeigen
   void kindPro_exec() const;

   //Anzeige der Projektionstypen bei tiled
   void tiled_exec() const;

   //stereoMode abhaengig vom typePro anzeigen
   void typePro_exec() const;

   //Front CheckBox immer aktivieren
   void frontChecked() const;

   //back wall auswaehlbar in Abhaengigkeit von expertMode
   void backEnable() const;

   //abhaengig vom stereo mode setzen der bothEyesCheckBox
   void stereo_exec() const;

   //Ein-/Ausblenden der Konfiguration fuer den Control Monitor
   void ctrlMon_exec() const;

   //Umschalten der Bilder fuer Monitor left/right of 3D
   void lrRB_exec() const;

   //Setzen der Variable expertMode
   void setExpertMode(const bool& changed);

};

#endif // VRCWPROJECTIONHW_H
