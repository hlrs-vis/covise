#ifndef VRCWPROJECTIONRESSIZE_H
#define VRCWPROJECTIONRESSIZE_H

#include <QWidget>
#include "ui_vrcwprojectionressize.h"

#include "vrcwutils.h"


class VRCWProjectionResSize : public QWidget
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWProjectionResSize(QWidget *parent = 0);
   ~VRCWProjectionResSize();


   /*****
    * functions
    *****/
   //Auslesen des GUI
   QVector<int> getGuiRes() const;
   aspRat getGuiAspectRatio() const;
   bool getGuiWallsSameResConf() const;
   QVector<int> getGuiScreenSize() const;
   QVector<int> getGuiCaveDim() const;

   //Setzen des GUI
   void setGuiAspRat(const aspRat& guiAR) const;
   void setGuiRes(const QVector<int>& guiRes) const;
   void setGuiScreenSize(const QVector<int>& guiScreenSize) const;
   void setGuiCaveDim(const QVector<int>& guiCaveDim) const;
   void setGuiCaveDimSbDisabled() const;

   //show/hide (CaveDimension && wallsSameResConfig) || screenSize
   void showCaveConfig(const bool& yes = true) const;

   //show Resolution, hide everything else
   void showCtrlMonConfig(const bool& yes = true) const;

   //enable/disable resolution and aspect ratio
   void enableRes(const bool& yes = true) const;

   //enable/disable screenSize
   void enableScreenSize(const bool& yes = true) const;

   //show/check or hide/uncheck calculate width/height combobox
   void showCalcWidthHeight(const bool yes = true) const;

   //check/uncheck calculate width/height combobox
   void checkCalcWidthHeight(const bool yes = true) const;


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWProjectionResSizeClass ui;


   /*****
    * functions
    *****/
   //show/hide user defined resolution spinbox
   void showUserRes(const bool& yes = true) const;

   //show/hide caveDim
   void showCaveDim(const bool& yes = true) const;

   //show/hide screenSize
   void showScreenSize(const bool& yes = true) const;

   //show/hide wallsSameResConfig
   void showWallsSameResConf(const bool& yes = true) const;


   /*****
    * variables
    *****/
   QMultiHash<aspRat, QString> resHash;
   //wird zur Steuerung der beteiligten Funktionen beim Aendern
   //der Resolution und des AspectRatio verwendet
   //Soll das mehrmalige Ausfuehren (Schleifenbildung) von Funktionen
   //verhindern
   bool aspRatChanged;


private slots:
   //show/hide user defined resolution spinbox
   void showUserRes_exec(const int& index);

   //aspect ratio changed and set resolution list and emit the signal
   void aspectRatioChanged(const bool& changed);

   //Aussenden der definierten Signale
   //
   //predefined or user defined resolution width or height
   void emitResValChanged();

   //wallsSameResConf
   void emitWallsResSameConfChanged();

   //calcWidthHeight
   void emitCalcWidthHeightChecked();

   //size width
   void emitSizeWidthValChanged(const int& newValue);

   //size height
   void emitSizeHeightValChanged(const int& newValue);

   //CaveDim width, depth or height
   void emitCaveDimValChanged();


signals:
   void aspRatValueChanged();
   void resValueChanged();
   void wallsResSameConfChanged();
   void sizeWidthValueChanged(const int& newValue);
   void sizeHeightValueChanged(const int& newValue);
   void calcWidthHeightChecked();
   void caveDimValueChanged();

};

#endif // VRCWPROJECTIONRESSIZE_H
