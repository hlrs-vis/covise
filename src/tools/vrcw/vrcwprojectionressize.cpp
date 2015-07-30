#include "vrcwprojectionressize.h"

#include <QStringBuilder>
#include <math.h>

#include "vrcwutils.h"


/*****
 * constructor - destructor
 *****/

VRCWProjectionResSize::VRCWProjectionResSize(QWidget *parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //Variablen setzen
   aspRatChanged = false;

   //don't show user defined resolution spinbox
   showUserRes(false);

   //don't show CaveDimension and wallsSameResConf
   //show screenSize
   showCaveConfig(false);

   //set the predefined resolutions
   //resHash.insert(_oth, "");
   //resHash.insert(_1610, "");
   //resHash.insert(_169, "");
   //resHash.insert(_43, "");
   //
   resHash.insert(_oth, "9334 x 7000");
   resHash.insert(_169, "7680 x 4320");
   resHash.insert(_oth, "4096 x 3112");
   resHash.insert(_43, "4096 x 3072");
   resHash.insert(_oth, "4096 x 2160");
   resHash.insert(_1610, "3840 x 2400");
   resHash.insert(_169, "3840 x 2160");
   resHash.insert(_43, "3200 x 2400");
   resHash.insert(_43, "2800 x 2100");
   resHash.insert(_oth, "2560 x 2048");
   resHash.insert(_169, "2560 x 1440");
   resHash.insert(_oth, "2048 x 1556");
   resHash.insert(_43, "2048 x 1536");
   resHash.insert(_169, "2048 x 1152");
   resHash.insert(_oth, "2048 x 1080");
   resHash.insert(_43, "1920 x 1440");
   resHash.insert(_1610, "1920 x 1200");
   resHash.insert(_169, "1920 x 1080");
   resHash.insert(_oth, "1920 x 1035");
   resHash.insert(_1610, "1680 x 1050");
   resHash.insert(_43, "1600 x 1200");
   resHash.insert(_169, "1600 x 900");
   resHash.insert(_43, "1536 x 1152");
   resHash.insert(_oth, "1440 x 1152");
   resHash.insert(_43, "1440 x 1080");
   resHash.insert(_1610, "1440 x 900");
   resHash.insert(_43, "1400 x 1050");
   resHash.insert(_oth, "1360 x 768");
   resHash.insert(_oth, "1280 x 1024");
   resHash.insert(_43, "1280 x 960");
   resHash.insert(_1610, "1280 x 800");
   resHash.insert(_169, "1280 x 720");
   resHash.insert(_43, "1200 x 900");
   resHash.insert(_43, "1152 x 864");
   resHash.insert(_43, "1024 x 768");
   resHash.insert(_169, "1024 x 576");
   resHash.insert(_43, "960 x 720");
   resHash.insert(_169, "960 x 540");
   resHash.insert(_43, "800 x 600");

   //set the predefined resolution list
   aspectRatioChanged(true);
}

VRCWProjectionResSize::~VRCWProjectionResSize()
{

}


/*****
 * public functions
 *****/

//Auslesen des GUI
//
QVector<int> VRCWProjectionResSize::getGuiRes() const
{
   QVector<int> guiRes(2,0);

   if (ui.preDefResComboBox->currentIndex() == 0)
   {
      guiRes[0] = ui.userResWidthSpinBox->value();
      guiRes[1] = ui.userResHeightSpinBox->value();
   }
   else
   {
      QString resStr = ui.preDefResComboBox->currentText();

      //RegExp: \D+ : Matches one or more non-digit
      QStringList resStrList = resStr.split(QRegExp("\\D+"),
            QString::SkipEmptyParts);

      //in CAVE mode it can happen that the string is empty or of size 1
      //because getGuiRes() is executed more than once. It is triggered with
      //signals from the CheckBox or aspectRatio-Buttons with
      //emitResValChanges or aspectRatioChanged() and there the
      //Checkbox is cleared and set.
      if (resStrList.size() == 2)
      {
         guiRes[0] = resStrList[0].toInt();
         guiRes[1] = resStrList[1].toInt();
      }
      //if (resStrList.size() != 2) then use the default guiRes(2,0)
   }

   return guiRes;
}

aspRat VRCWProjectionResSize::getGuiAspectRatio() const
{
   aspRat aR;

   if (ui.aspRat4to3RadioButton->isChecked())
   {
      aR = _43;
   }
   else if (ui.aspRat16to9RadioButton->isChecked())
   {
      aR = _169;
   }
   else if (ui.aspRat16to10RadioButton->isChecked())
   {
      aR = _1610;
   }
   else if (ui.aspRatOtherRadioButton->isChecked())
   {
      aR = _oth;
   }
   else
   {
      aR = _all;
   }

   return aR;
}

bool VRCWProjectionResSize::getGuiWallsSameResConf() const
{
   return ui.wallsSameResConfCheckBox->isChecked();
}

QVector<int> VRCWProjectionResSize::getGuiScreenSize() const
{
   QVector<int> guiScreenSize(2,0);

   guiScreenSize[0] = ui.screenSizeWidthSpinBox->value();
   guiScreenSize[1] = ui.screenSizeHeightSpinBox->value();

   return guiScreenSize;
}

QVector<int> VRCWProjectionResSize::getGuiCaveDim() const
{
   QVector<int> guiCaveDim(3,0);

   guiCaveDim[0] = ui.caveDimWidthSpinBox->value();
   guiCaveDim[1] = ui.caveDimDepthSpinBox->value();
   guiCaveDim[2] = ui.caveDimHeightSpinBox->value();

   return guiCaveDim;
}

//Setzen des GUI
//
void VRCWProjectionResSize::setGuiAspRat(const aspRat& guiAR) const
{
   switch (guiAR)
   {
      case _43:
      {
         ui.aspRat4to3RadioButton->setChecked(true);
         break;
      }
      case _169:
      {
         ui.aspRat16to9RadioButton->setChecked(true);
         break;
      }
      case _1610:
      {
         ui.aspRat16to10RadioButton->setChecked(true);
         break;
      }
      case _oth:
      {
         ui.aspRatOtherRadioButton->setChecked(true);
         break;
      }
      case _all:
      {
         ui.aspRatAllRadioButton->setChecked(true);
         break;
      }
   }
}

void VRCWProjectionResSize::setGuiRes(const QVector<int>& guiRes) const
{
   int width = guiRes[0];
   int height = guiRes[1];
   QString resStr = QString::number(width) % " x " % QString::number(height);
   int index = ui.preDefResComboBox->findText(resStr);

   if (index > 0)
   {
      ui.preDefResComboBox->setCurrentIndex(index);
   }
   else
   {
      ui.preDefResComboBox->setCurrentIndex(0);
      ui.userResWidthSpinBox->setValue(width);
      ui.userResHeightSpinBox->setValue(height);
   }
}

void VRCWProjectionResSize::setGuiScreenSize
   (const QVector<int>& guiScreenSize) const
{
   ui.screenSizeWidthSpinBox->setValue(guiScreenSize[0]);
   ui.screenSizeHeightSpinBox->setValue(guiScreenSize[1]);
}

void VRCWProjectionResSize::setGuiCaveDim(const QVector<int>& guiCaveDim) const
{
   ui.caveDimWidthSpinBox->setValue(guiCaveDim[0]);
   ui.caveDimDepthSpinBox->setValue(guiCaveDim[1]);
   ui.caveDimHeightSpinBox->setValue(guiCaveDim[2]);
}

//enable/disable the spinboxes for cave width, depth, height
//
void VRCWProjectionResSize::setGuiCaveDimSbDisabled() const
{
   QVector<int> caveDimVal = getGuiCaveDim();

   for (int i = 0; i < caveDimVal.size(); ++i)
   {
      //placeholder variable
      QSpinBox* qsp;

      switch (i)
      {
         case 0:
         {
            qsp = ui.caveDimWidthSpinBox;
            break;
         }
         case 1:
         {
            qsp = ui.caveDimDepthSpinBox;
            break;
         }
         case 2:
         {
            qsp = ui.caveDimHeightSpinBox;
            break;
         }
         default:
         {
            //do nothing
            break;
         }
      }

      if (caveDimVal[i] != 0)
      {
         qsp->setEnabled(false);
      }
      else
      {
         qsp->setEnabled(true);
      }
   }
}

//show/hide (CaveDimension && wallsSameResConfig) || screenSize
//
void VRCWProjectionResSize::showCaveConfig(const bool& yes) const
{
   if (yes)
   {
      showCaveDim();
      showWallsSameResConf();
      showScreenSize(false);
   }
   else
   {
      showCaveDim(false);
      showWallsSameResConf(false);
      showScreenSize();
   }
}

//show only the resolution, hide everything else
//
void VRCWProjectionResSize::showCtrlMonConfig(const bool& yes) const
{
   if (yes)
   {
      showCaveDim(false);
      showWallsSameResConf(false);
      showScreenSize(false);
   }
}

//enable/disable resolution and aspect ratio
//
void VRCWProjectionResSize::enableRes(const bool& yes) const
{
   if (yes)
   {
      //aspect ratio
      ui.aspectRatioWidget->setEnabled(true);
      ui.aspectRatioLabel->setEnabled(true);
      //resolution
      ui.resWidget->setEnabled(true);
      ui.resLabel->setEnabled(true);
   }
   else
   {
      //aspect ratio
      ui.aspectRatioWidget->setEnabled(false);
      ui.aspectRatioLabel->setEnabled(false);
      //resolution
      ui.resWidget->setEnabled(false);
      ui.resLabel->setEnabled(false);
   }
}

//enable/disable screenSize
//
void VRCWProjectionResSize::enableScreenSize(const bool& yes) const
{
   if (yes)
   {
      ui.screenSizeCalcWidget->setEnabled(true);
      ui.screenSizeLabel->setEnabled(true);
   }
   else
   {
      ui.screenSizeCalcWidget->setEnabled(false);
      ui.screenSizeLabel->setEnabled(false);
   }
}

//show/check or hide/uncheck calculate width/height combobox
//
void VRCWProjectionResSize::showCalcWidthHeight(const bool yes) const
{
   if (yes)
   {
      ui.calcWidthHeightCheckBox->show();
      ui.calcWidthHeightCheckBox->setChecked(true);
   }
   else
   {
      ui.calcWidthHeightCheckBox->hide();
      ui.calcWidthHeightCheckBox->setChecked(false);
   }
}

//check/uncheck calculate width/height combobox
//
void VRCWProjectionResSize::checkCalcWidthHeight(const bool yes) const
{
   //all checkboxes that are not visible because their tab aren't shown
   //at the moment will be unchecked and
   //the checkbox that is seen will be checked
   if (yes && ui.calcWidthHeightCheckBox->isVisible())
   {
      ui.calcWidthHeightCheckBox->setChecked(true);
   }
   else if (!yes && !ui.calcWidthHeightCheckBox->isVisible())
   {
      ui.calcWidthHeightCheckBox->setChecked(false);
   }
}


/*****
 * private functions
 *****/

//show/hide user defined resolution spinbox
//
void VRCWProjectionResSize::showUserRes(const bool& yes) const
{
   if (yes)
   {
      ui.userResWidget->show();
   }
   else
   {
      ui.userResWidget->hide();
   }
}

//show/hide caveDim
//
void VRCWProjectionResSize::showCaveDim(const bool& yes) const
{
   if (yes)
   {
      ui.caveDimWidget->show();
      ui.caveDimLabel->show();
      ui.caveDimEdgesPictureLabel->show();
   }
   else
   {
      ui.caveDimWidget->hide();
      ui.caveDimLabel->hide();
      ui.caveDimEdgesPictureLabel->hide();
   }
}

//show/hide screenSize
//
void VRCWProjectionResSize::showScreenSize(const bool& yes) const
{
   if (yes)
   {
      ui.screenSizeCalcWidget->show();
      ui.screenSizeLabel->show();
   }
   else
   {
      ui.screenSizeCalcWidget->hide();
      ui.screenSizeLabel->hide();
   }
}

//show/hide wallsSameResConfig
//
void VRCWProjectionResSize::showWallsSameResConf(const bool& yes) const
{
   if (yes)
   {
      ui.wallsSameResConfCheckBox->show();
   }
   else
   {
      ui.wallsSameResConfCheckBox->hide();
   }
}


/*****
 * private slots
 *****/

//show/hide user defined resolution spinbox
//
void VRCWProjectionResSize::showUserRes_exec(const int& index)
{
   //aspRatChanged soll verhindern, dass die Funktion wie eine Schleife
   //immer und immer wieder ausgefuehrt wird
   //ohne wird das AspectRatio immer auf _all bleiben
   if (aspRatChanged == false)
   {
      if (index == 0)
      {
         //if --other-- resolution is selected the aspectRatio should be all
         //and there --other-- and the input fields for
         //user defined resolution visible
         setGuiAspRat(_all);

         //verhindert fuer die comboBox, dass diese Funktion mehrfach
         //ausgefuehrt wird
         aspRatChanged = true;
         ui.preDefResComboBox->setCurrentIndex(0);
         aspRatChanged = false;

         showUserRes();
      }
      else
      {
         showUserRes(false);
      }
   }
}

//aspect ratio changed and set resolution list
//
void VRCWProjectionResSize::aspectRatioChanged(const bool& changed)
{
   if (changed)
   {
      int index;

      //determine, which radio button is checked
      aspRat aR = getGuiAspectRatio();

      //*****
      //**
      aspRatChanged = true;

      //set redefined resolution comboBox for specified aspectRatio
      ui.preDefResComboBox->clear();
      ui.preDefResComboBox->addItem(OTHER);
      if (aR != _all)
      {
         ui.preDefResComboBox->addItems(resHash.values(aR));
      }
      else
      {
         ui.preDefResComboBox->addItems(resHash.values());
      }

      aspRatChanged = false;
      //**
      //*****

      //set default value for selected aspectRatio
      switch (aR)
      {
         case _43:
         {
            index = ui.preDefResComboBox->findText("1600 x 1200");
            ui.preDefResComboBox->setCurrentIndex(index);
            break;
         }
         case _169:
         {
            index = ui.preDefResComboBox->findText("1920 x 1080");
            ui.preDefResComboBox->setCurrentIndex(index);
            break;
         }
         case _1610:
         {
            index = ui.preDefResComboBox->findText("1920 x 1200");
            ui.preDefResComboBox->setCurrentIndex(index);
            break;
         }
         case _oth:
         {
            index = ui.preDefResComboBox->findText("1280 x 1024");
            ui.preDefResComboBox->setCurrentIndex(index);
            break;
         }
         case _all:
         {
            ui.preDefResComboBox->setCurrentIndex(1);
            break;
         }
      }

      //emit the signal
      emit aspRatValueChanged();
   }
}

//Aussenden der definierten Signale
//
//predefined or user defined resolution width or height
//
void VRCWProjectionResSize::emitResValChanged()
{
   if (ui.preDefResComboBox->hasFocus() ||
         ui.userResHeightSpinBox->hasFocus() ||
         ui.userResWidthSpinBox->hasFocus())
   {
      emit resValueChanged();
   }
}

//wallsSameResConf
//
void VRCWProjectionResSize::emitWallsResSameConfChanged()
{
   emit wallsResSameConfChanged();
}

//calcWidthHeight
void VRCWProjectionResSize::emitCalcWidthHeightChecked()
{
   emit calcWidthHeightChecked();
}


//size width
//
void VRCWProjectionResSize::emitSizeWidthValChanged(const int& newValue)
{
   //calculate the height for newValue (width) depending on aspect ratio
   if (ui.calcWidthHeightCheckBox->isChecked())
   {
      QVector<int> res = getGuiRes();
      int newHeight = round(newValue * res[1] * 1.0 / res[0]);

      //set the new height only if the HeightSpinBox hasn't the focus
      if (!ui.screenSizeHeightSpinBox->hasFocus())
      {
         ui.screenSizeHeightSpinBox->setValue(newHeight);
      }
   }

   emit sizeWidthValueChanged(newValue);
}

//size height
//
void VRCWProjectionResSize::emitSizeHeightValChanged(const int& newValue)
{
   //calculate the width for newValue (height) depending on aspect ratio
   if (ui.calcWidthHeightCheckBox->isChecked())
   {
      QVector<int> res = getGuiRes();
      int newWidth = round(newValue * res[0] * 1.0 / res[1]);

      //set the new width only if the WidthSpinBox hasn't the focus
      if (!ui.screenSizeWidthSpinBox->hasFocus())
      {
         ui.screenSizeWidthSpinBox->setValue(newWidth);
      }
   }

   emit sizeHeightValueChanged(newValue);
}

//CaveDim width, depth or height
//
void VRCWProjectionResSize::emitCaveDimValChanged()
{
   emit caveDimValueChanged();
}
