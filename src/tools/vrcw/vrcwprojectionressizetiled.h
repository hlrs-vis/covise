#ifndef VRCWPROJECTIONRESSIZETILED_H
#define VRCWPROJECTIONRESSIZETILED_H

#include <QWidget>
#include "ui_vrcwprojectionressizetiled.h"

#include "vrcwutils.h"

class VRCWProjectionResSize;


class VRCWProjectionResSizeTiled: public QWidget
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWProjectionResSizeTiled(QWidget *parent = 0);
   ~VRCWProjectionResSizeTiled();


   /*****
    * functions
    *****/
   //Auslesen des GUI
   QVector<int> getGuiRowCol() const;
   typePro getGuiTypeProj() const;
   QVector<int> getGuiRes() const;
   aspRat getGuiAspectRatio() const;
   QVector<int> getGuiScreenSize() const;
   QVector<int> getGuiWallSize() const;
   QVector<int> getGuiOverlap() const;
   QVector<int> getGuiFrame() const;


   //Setzen des GUI
   void setGuiAspRat(const aspRat& guiAR) const;
   void setGuiRes(const QVector<int>& guiRes) const;
   void setGuiWallSize(const QVector<int>& guiWallSize) const;

   //Enable/disable Resolution
   void enableRes(const bool& yes = true) const;

   //show/check or hide/uncheck calculate width/height combobox
   void showCalcWidthHeight(const bool yes = true) const;

   //check/uncheck calculate width/height combobox
   void checkCalcWidthHeight(const bool yes = true) const;

   //Set kind of projection
   void setKindProjection(const proKind& kind);

   //set type of projection
   void setTypeProjection(const typePro& typeP);

   //set aspectRatio
   void setAspRat(const aspRat&  aR);


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWProjectionResSizeTiledClass ui;


   /*****
    * functions
    *****/
   //show/hide overlap radioButton and widget
   void showOverlap(const bool& yes = true) const;

   //show/hide frame radioButton and widget
   void showFrame(const bool& yes = true) const;


   /*****
    * variables
    *****/
   VRCWProjectionResSize* projectionResSize;

   //Value from DimPowerwall or DimCave
   proKind kindData;
   typePro typePData;


private slots:
   //check displayRadioButton or projectorRadioButton
   //and set the appropriate things
   void checkDispProjRB() const;

   //check wallSizeRadioButton
   void checkWallRB() const;

   //check overlap or frame RB
   void checkOverlapFrameRB() const;

   //enable/disable horizontal/vertical overlap
   //depending on numbers of row/column
   void enableHoriVertOverlap() const;

   //calculate the values of the disabled dimensions
   void calcDisabledDims() const;

   //calcWidthHeight
   void emitCalcWidthHeightChecked();

   //wallSize width
   void emitWallSizeWidthValChanged(const int& newValue);

   //wallSize height
   void emitWallSizeHeightValChanged(const int& newValue);


signals:
   void calcWidthHeightChecked();
   void wallSizeWidthValueChanged(const int& newValue);
   void wallSizeHeightValueChanged(const int& newValue);

};

#endif // VRCWPROJECTIONRESSIZETILED_H
