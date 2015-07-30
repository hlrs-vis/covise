#ifndef VRCWPROJECTIONDIMCAVE_H
#define VRCWPROJECTIONDIMCAVE_H

#include <QWidget>
#include "ui_vrcwprojectiondimcave.h"

#include "vrcwbase.h"
#include "vrcwutils.h"

class VRCWProjectionResSize;
class VRCWProjectionResSizeTiled;
class VRCWProjectionVposFloor;
class wallVal;


class VRCWProjectionDimCave : public QWidget, public VRCWBase
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWProjectionDimCave(QWidget *parent = 0);
   ~VRCWProjectionDimCave();


   /*****
    * functions
    *****/
   //Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
   //Bearbeitung und Ueberpruefung der Eingaben im GUI
   int processGuiInput(const int& index, const QList<VRCWBase*>& vrcwList);

   //Anzeigen der zu konfigurierenden Waende
   void setGuiWallsToConfigure(const QList<cWall>& walls);

   //Entgegennehmen der Parameter von ProjectionHw fuer die Bestimmung
   //der Projection
   void setProjectionHwParams(const proKind& kind, const bool& tiled,
         const typePro& typeP, const stType& stereo, const bool& bothEyes,
         const int& graka);


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWProjectionDimCaveClass ui;


   /*****
    * functions
    *****/
   //set the aspect ratio
   //from general in selected tabs depending on wallSameResConf checkBox
   void setWallsAspRat() const;

   //set the predefined or user defined resolution
   //from general in selected tabs depending on wallSameResConf checkBox
   void setWallsResWH() const;

   //auf vorhandenen Tabs die predefined resolution und user defined width und
   //height aktivieren/deaktivieren abhaengig von wallSameResConfCheckBox
   void wallsResEnable(const bool& yes = true) const;

   //setzen der ScreenSize-Werte auf den Tabs
   void setWallsSizeWH() const;

   //setzen des zero point
   void setZeroPoint(const zPoint& zp);

   //show/hide tiledDims or untiledDims
   void showTiledDim(const bool& yes = true) const;

   // Auslesen des GUI
   //general
   QVector<int> getGuiResGeneral() const;
   aspRat getGuiAspectRatioGeneral() const;
   QVector<int> getGuiCaveDim() const;
   bool getGuiWallsSameResConf() const;
   QList<wallVal*> getGuiCaveWallDim() const;
   zPoint getGuiZeroPoint() const;

   //Erzeugen der Liste der Projektionen
   QStringList createProjection();


   /*****
    * variables
    *****/
   //general Page
   VRCWProjectionResSize* generalDim;
   VRCWProjectionVposFloor* generalPos;
   //front Page
   VRCWProjectionResSize* front;
   VRCWProjectionResSizeTiled* frontTiled;
   //left Page
   VRCWProjectionResSize* left;
   VRCWProjectionResSizeTiled* leftTiled;
   //right Page
   VRCWProjectionResSize* right;
   VRCWProjectionResSizeTiled* rightTiled;
   //bottom Page
   VRCWProjectionResSize* bottom;
   VRCWProjectionResSizeTiled* bottomTiled;
   //top Page
   VRCWProjectionResSize* top;
   VRCWProjectionResSizeTiled* topTiled;
   //back Page
   VRCWProjectionResSize* back;
   VRCWProjectionResSizeTiled* backTiled;

   //Values
   //Liste der zu konfigurierenden Waende
   QList<cWall> wallsToConfigure;
   //general
   //caveDim holds the width, depth and height of the cave
   //with and height are values of front and back wall
   //the cave depth depends on left, right, top, bottom
   //all of them can have different widths/heights (is the cave depth)
   //for cave depth the largest value of them is decisive
   //front and back are both connected to the largest value
   //left, right, bottom, top are connected to the front wall
   QVector<int> caveDim;
   //Walls
   QList<wallVal*> caveWallDim;
   //store old value
   QString typePOld;

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
   enum cMode {Cfpf, Cfpt, Cfat, Ctpf, Ctpt, Ctat};


private slots:
   //set the aspect ratio
   //from general in selected tabs depending on wallSameResConf checkBox
   void setWallsAspRat_exec() const;

   //set the predefined oder user defined resolution
   //from general in selected tabs depending on wallSameResConf checkBox
   void setWallsResWH_exec() const;

   //enable or disable the predefined resolution
   //and/or user defined width/height spinbox
   void wallsResEnable_exec() const;

   //setzen der Werte aus caveDim general in Size in den vorhandenen Tabs
   void setWallsSizeWH_exec();

   //uncheck the calcWidthHeightCheckbox on the tabs
   void uncheckCalcWidthHeightCheckbox() const;

   //Entgegennehmen des size Wertes fuer CaveDim width und setzen auf den Tabs
   void setCaveDimWidthWall_exec(const int& newValue);

   //Entgegennehmen des size Wertes fuer CaveDim depth und setzen auf den Tabs
   void setCaveDimDepthWall_exec(const int& newValue);

   //Entgegennehmen des size Wertes fuer CaveDim height und setzen auf den Tabs
   void setCaveDimHeightWall_exec(const int& newValue);

   //location of zero point changed and set description
   void zeroPointChanged();

};

#endif // VRCWPROJECTIONDIMCAVE_H
