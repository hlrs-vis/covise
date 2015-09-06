#include "vrcwprojectionressizetiled.h"

#include <QDebug>

#include "vrcwprojectionressize.h"


/*****
 * constructor - destructor
 *****/

VRCWProjectionResSizeTiled::VRCWProjectionResSizeTiled(QWidget *parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //GUI
   //
   projectionResSize = new VRCWProjectionResSize(ui.resSizeOneProjectWidget);
   ui.resSizeOneProjectWidgetGLayout->addWidget(projectionResSize);

   //hide frame
   showFrame(false);

   //enable wallSize / disable overlap
   checkWallRB();

   //set variable
   //Powerwall is also default for CAVE,
   //VRCWProjectionDimCave do not set the variable kindData
   kindData = Powerwall;
   typePData = Projector;

   //show type of projection and disable interaction
   ui.displayRadioButton->setEnabled(false);
   ui.projectorRadioButton->setEnabled(false);

   //Signals and Slots
   //
   //Empfang der Signale aus projectionResSize
   // -um die Werte der wallSize, overlap oder frame zu aendern
   connect(projectionResSize, SIGNAL(sizeWidthValueChanged(int)), this,
         SLOT(calcDisabledDims()));
   connect(projectionResSize, SIGNAL(sizeHeightValueChanged(int)), this,
         SLOT(calcDisabledDims()));
   // -um das Signal calcWidthHeightChecked() an VRCWProjectionDimCave
   //  weiter zu reichen
   connect(projectionResSize, SIGNAL(calcWidthHeightChecked()), this,
            SLOT(emitCalcWidthHeightChecked()));
}

VRCWProjectionResSizeTiled::~VRCWProjectionResSizeTiled()
{

}


/*****
 * functions
 *****/

//Auslesen des GUI
//
QVector<int> VRCWProjectionResSizeTiled::getGuiRowCol() const
{
   QVector<int> guiRowCol(2,0);
   guiRowCol[0] = ui.tiledRowSpinBox->value();
   guiRowCol[1] = ui.tiledColSpinBox->value();
   return guiRowCol;
}

typePro VRCWProjectionResSizeTiled::getGuiTypeProj() const
{
   typePro typeProj;

   if (ui.displayRadioButton->isChecked())
   {
      typeProj = Monitor;
   }
   else
   {
      typeProj = Projector;
   }

   return typeProj;
}

QVector<int> VRCWProjectionResSizeTiled::getGuiRes() const
{
   return projectionResSize->getGuiRes();
}

aspRat VRCWProjectionResSizeTiled::getGuiAspectRatio() const
{
   return projectionResSize->getGuiAspectRatio();
}

QVector<int> VRCWProjectionResSizeTiled::getGuiScreenSize() const
{
   return projectionResSize->getGuiScreenSize();
}

QVector<int> VRCWProjectionResSizeTiled::getGuiWallSize() const
{
   QVector<int> wallSize(2,0);
   wallSize[0] = ui.wallSizeWidthSpinBox->value();
   wallSize[1] = ui.wallSizeHeightSpinBox->value();
   return wallSize;
}

QVector<int> VRCWProjectionResSizeTiled::getGuiOverlap() const
{
   QVector<int> overlap(2,0);
   overlap[0] = ui.overlapHorizontalSpinBox->value();
   overlap[1] = ui.overlapVerticalSpinBox->value();
   return overlap;
}

QVector<int> VRCWProjectionResSizeTiled::getGuiFrame() const
{
   QVector<int> frame(2,0);
   frame[0] = ui.frameLRSpinBox->value();
   frame[1] = ui.frameBTSpinBox->value();
   return frame;
}

//Setzen des GUI
//
void VRCWProjectionResSizeTiled::setGuiAspRat(const aspRat& guiAR) const
{
   projectionResSize->setGuiAspRat(guiAR);
}

void VRCWProjectionResSizeTiled::setGuiRes(const QVector<int>& guiRes) const
{
   projectionResSize->setGuiRes(guiRes);
}

void VRCWProjectionResSizeTiled::setGuiWallSize
   (const QVector<int>& guiWallSize) const
{
   ui.wallSizeWidthSpinBox->setValue(guiWallSize[0]);
   ui.wallSizeHeightSpinBox->setValue(guiWallSize[1]);
}

//Enable/disable Resolution
//
void VRCWProjectionResSizeTiled::enableRes(const bool& yes) const
{
   projectionResSize->enableRes(yes);
}

//show/check or hide/uncheck calculate width/height combobox
//
void VRCWProjectionResSizeTiled::showCalcWidthHeight(const bool yes) const
{
   projectionResSize->showCalcWidthHeight(yes);
}

//check/uncheck calculate width/height combobox
//
void VRCWProjectionResSizeTiled::checkCalcWidthHeight(const bool yes) const
{
   projectionResSize->checkCalcWidthHeight(yes);
}

//Set kind of projection
//
void VRCWProjectionResSizeTiled::setKindProjection(const proKind& kind)
{
   //this function is only called by VRCWProjectionDimPowerwall
   //VRCWProjection do not call this function
   //CAVE use the the default Powerwall like defined in constructor

   kindData = kind;
}

//set type of projection
void VRCWProjectionResSizeTiled::setTypeProjection(const typePro& typeP)
{
   typePData = typeP;

   //set display or projector and default AspectRatio
   switch(typePData)
   {
      case Monitor:
      {
         ui.displayRadioButton->setChecked(true);
         break;
      }
      case Projector:
      {
         ui.projectorRadioButton->setChecked(true);
         break;
      }
   }
}

//set aspectRatio
void VRCWProjectionResSizeTiled::setAspRat(const aspRat&  aR)
{
   projectionResSize->setGuiAspRat(aR);
}


/*****
 * private functions
 *****/

//show/hide overlap radioButton, widget and description picture
//
void VRCWProjectionResSizeTiled::showOverlap(const bool& yes) const
{
   if (yes)
   {
      ui.overlapRadioButton->show();
      ui.overlapWidget->show();
      ui.tiledProjectorPictLabel->show();
   }
   else
   {
      ui.overlapRadioButton->hide();
      ui.overlapWidget->hide();
      ui.tiledProjectorPictLabel->hide();
   }
}

//show/hide frame radioButton, widget and description picture
//
void VRCWProjectionResSizeTiled::showFrame(const bool& yes) const
{
   if (yes)
   {
      ui.frameRadioButton->show();
      ui.frameWidget->show();
      ui.tiledDisplayPictLabel->show();
   }
   else
   {
      ui.frameRadioButton->hide();
      ui.frameWidget->hide();
      ui.tiledDisplayPictLabel->hide();
   }
}


/*****
 * private slots
 *****/

//check displayRadioButton or projectorRadioButton
//and set the appropriate things
//
void VRCWProjectionResSizeTiled::checkDispProjRB() const
{
   if (ui.displayRadioButton->isChecked())
   {
      //show frame
      showFrame();

      //hide overlap
      showOverlap(false);
   }
   else
   {
      //show overlap
      showOverlap();

      //hide frame
      showFrame(false);
   }

   //calculate the values of the disabled dimensions
   calcDisabledDims();
}

//check wallSizeRadioButton
//
void VRCWProjectionResSizeTiled::checkWallRB() const
{
   //check/uncheck radioButtons
   ui.wallSizeRadioButton->setChecked(true);
   ui.overlapRadioButton->setChecked(false);
   ui.frameRadioButton->setChecked(false);

   //disable horizontal/vertical overlap
   ui.overlapHorizontalSpinBox->setEnabled(false);
   ui.overlapHorizontalLabel->setEnabled(false);
   ui.overlapVerticalSpinBox->setEnabled(false);
   ui.overlapVerticalLabel->setEnabled(false);

   //disable LR/BT frame
   ui.frameLRSpinBox->setEnabled(false);
   ui.frameLRLabel->setEnabled(false);
   ui.frameBTSpinBox->setEnabled(false);
   ui.frameBTLabel->setEnabled(false);

   //enable width/height of wallSize
   ui.wallSizeHeightSpinBox->setEnabled(true);
   ui.wallSizeHeightLabel->setEnabled(true);
   ui.wallSizeWidthSpinBox->setEnabled(true);
   ui.wallSizeWidthLabel->setEnabled(true);

   //enable width/height of ScreenSize
   projectionResSize->enableScreenSize(true);
}

//check overlap or frame RB
//
void VRCWProjectionResSizeTiled::checkOverlapFrameRB() const
{
   //check/uncheck radioButtons
   ui.wallSizeRadioButton->setChecked(false);
   ui.overlapRadioButton->setChecked(true);
   ui.frameRadioButton->setChecked(true);

   //enable horizontal/vertical overlap
   //depending on numbers of row/column
   enableHoriVertOverlap();

   //enable LR/BT frame
   ui.frameLRSpinBox->setEnabled(true);
   ui.frameLRLabel->setEnabled(true);
   ui.frameBTSpinBox->setEnabled(true);
   ui.frameBTLabel->setEnabled(true);

   switch (kindData)
   {
      case CAVE:
      {
         //do nothing,
         //because kindData is not set to CAVE
         //instead kindData = Powerwall is used
         break;
      }
      case Powerwall:
      {
         //enable width/height of ScreenSize
         projectionResSize->enableScreenSize(true);

        //disable width/height of wallSize
         ui.wallSizeHeightSpinBox->setEnabled(false);
         ui.wallSizeHeightLabel->setEnabled(false);
         ui.wallSizeWidthSpinBox->setEnabled(false);
         ui.wallSizeWidthLabel->setEnabled(false);
         break;
      }
      case _3D_TV://tiled is not available for _3D_TV
      {
         //do nothing
         break;
      }
   }
}

//enable/disable horizontal/vertical overlap
//depending on numbers of row/column
//
void VRCWProjectionResSizeTiled::enableHoriVertOverlap() const
{
   QVector<int> rowCol = getGuiRowCol();

   if (ui.overlapRadioButton->isChecked())
   {
      if (rowCol[0] > 1)
      {
         ui.overlapVerticalSpinBox->setEnabled(true);
         ui.overlapVerticalLabel->setEnabled(true);
      }
      else
      {
         ui.overlapVerticalSpinBox->setEnabled(false);
         ui.overlapVerticalLabel->setEnabled(false);
      }

      if (rowCol[1] > 1)
      {
         ui.overlapHorizontalSpinBox->setEnabled(true);
         ui.overlapHorizontalLabel->setEnabled(true);
      }
      else
      {
         ui.overlapHorizontalSpinBox->setEnabled(false);
         ui.overlapHorizontalLabel->setEnabled(false);
      }
   }

   calcDisabledDims();
}

//calculate the values of the disabled dimensions
//
void VRCWProjectionResSizeTiled::calcDisabledDims() const
{
   QVector<int> rowCol = getGuiRowCol();
   QVector<int> screenSize = projectionResSize->getGuiScreenSize();
   QVector<int> wallSize = getGuiWallSize();
   QVector<int> overlap = getGuiOverlap();
   QVector<int> frame = getGuiFrame();


   qDebug() << "row = " << rowCol[0];
   qDebug() << "col = " << rowCol[1];
   qDebug() << "wallSize[0] = " << wallSize[0];
   qDebug() << "wallSize[1] = " << wallSize[1];
   qDebug() << "screenSize[0] = " << screenSize[0];
   qDebug() << "screenSize[1] = " << screenSize[1];
   qDebug() << "overlap[0] = " << overlap[0];
   qDebug() << "overlap[1] = " << overlap[1];
   qDebug() << "frame[0] = " << frame[0];
   qDebug() << "frame[1] = " << frame[1];


   if (ui.displayRadioButton->isChecked())
   {
      if (ui.wallSizeRadioButton->isChecked())
      {
         //left/right frame
         int lr = (wallSize[0] / rowCol[1]  - screenSize[0]) / 2;

         qDebug() << "lr = " << lr;

         ui.frameLRSpinBox->setValue(lr);

         //bottom/top frame
         int bt = (wallSize[1] / rowCol[0] - screenSize[1]) / 2;

         qDebug() << "bt = " << bt;

         ui.frameBTSpinBox->setValue(bt);

         qDebug() << "";
      }
      else
      {
         switch (kindData)
         {
            case CAVE:
            {
               //do nothing,
               //because kindData is not set to CAVE
               //instead kindData = Powerwall is used
               break;
            }
            case Powerwall:
            {
               //wallSize width
               int ww = (frame[0] * 2 + screenSize[0]) * rowCol[1];

               qDebug() << "wallSize width = " << ww;

               ui.wallSizeWidthSpinBox->setValue(ww);

               //wallSize height
               int wh = (frame[1] * 2 + screenSize[1]) * rowCol[0];

               qDebug() << "wallSize height = " << wh;

               ui.wallSizeHeightSpinBox->setValue(wh);

               qDebug() << "";
               break;
            }
            case _3D_TV://tiled is not available for _3D_TV
            {
               //do nothing
               break;
            }
         }
      }
   }
   else if (ui.projectorRadioButton->isChecked())
   {
      if (ui.wallSizeRadioButton->isChecked())
      {
         //horizontal overlap
         int ho = 0;

         if (rowCol[1] > 1)
         {
            ho = (screenSize[0] * rowCol[1] - wallSize[0]) / (rowCol[1] - 1);
            if (ho > wallSize[0])
            {
               ho = wallSize[0];
            }
         }

         qDebug() << "horizontal overlap = " << ho;

         ui.overlapHorizontalSpinBox->setValue(ho);

         //vertical overlap
         int vo = 0;

         if (rowCol[0] > 1)
         {
            vo = (screenSize[1] * rowCol[0] - wallSize[1]) / (rowCol[0] - 1);
            if (vo > wallSize[1])
            {
               vo = wallSize[1];
            }
         }

         qDebug() << "vertical overlap = " << vo;

         ui.overlapVerticalSpinBox->setValue(vo);

         qDebug() << "";
      }
      else
      {
         switch (kindData)
         {
            case CAVE:
            {
               //do nothing,
               //because kindData is not set to CAVE
               //instead kindData = Powerwall is used
               break;
            }
            case Powerwall:
            {
               //wallSize width
               int ww = screenSize[0] * rowCol[1] - overlap[0] *
                     (rowCol[1] - 1);

               qDebug() << "wallSize width = " << ww;

               ui.wallSizeWidthSpinBox->setValue(ww);

               //wallSize height
               int wh = screenSize[1] * rowCol[0] - overlap[1] *
                     (rowCol[0] - 1);

               qDebug() << "wallSize height = " << wh;

               ui.wallSizeHeightSpinBox->setValue(wh);

               qDebug() << "";
               break;
            }
            case _3D_TV://tiled is not available for _3D_TV
            {
               //do nothing
               break;
            }
         }
      }
   }
}

//calcWidthHeight
//
void VRCWProjectionResSizeTiled::emitCalcWidthHeightChecked()
{
   emit calcWidthHeightChecked();
}

//wallSize width
//
void VRCWProjectionResSizeTiled::emitWallSizeWidthValChanged
   (const int& newValue)
{
   emit wallSizeWidthValueChanged(newValue);
   calcDisabledDims();
}

//wallSize height
//
void VRCWProjectionResSizeTiled::emitWallSizeHeightValChanged
   (const int& newValue)
{
   emit wallSizeHeightValueChanged(newValue);
   calcDisabledDims();
}
