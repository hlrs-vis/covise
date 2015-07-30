#include "vrcwprojectionvposfloor.h"


/*****
 * constructor - destructor
 *****/

VRCWProjectionVposFloor::VRCWProjectionVposFloor(QWidget *parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //don't show caveHint, show pwHint
   showCaveHint(false);
}

VRCWProjectionVposFloor::~VRCWProjectionVposFloor()
{

}


/*****
 * public functions
 *****/

// Auslesen des GUI
//
QVector<int> VRCWProjectionVposFloor::getGuiVPos() const
{
   QVector<int> guiVPos(3);

   guiVPos[0] = ui.vPosXSpinBox->value();
   guiVPos[1] = ui.vPosYSpinBox->value();
   guiVPos[2] = ui.vPosZSpinBox->value();

   return guiVPos;
}

int VRCWProjectionVposFloor::getGuiFloor() const
{
   return ui.floorSpinBox->value();
}

//Setzen des GUI
//
void VRCWProjectionVposFloor::setGuiVPos(const QVector<int>& guiVPos) const
{
   ui.vPosXSpinBox->setValue(guiVPos[0]);
   ui.vPosYSpinBox->setValue(guiVPos[1]);
   ui.vPosZSpinBox->setValue(guiVPos[2]);
}

void VRCWProjectionVposFloor::setGuiFloor(const int& guiFloor) const
{
   ui.floorSpinBox->setValue(guiFloor);
}

void VRCWProjectionVposFloor::showCaveHint(const bool& yes) const
{
   if (yes)
   {
      setCaveHint();
   }
   else
   {
      setPwHint();
   }
}


/*****
 * private functions
 *****/

//set powerwall hint
//
void VRCWProjectionVposFloor::setPwHint() const
{
   ui.pwCaveHintLabel->setText("The zero point is located in the middle "
         "of the screen.\n"
         "The viewer position and floor height are measured from the "
         "zero point in the direction of the coordinate axis.\n"
         "The viewer position is the position of the eyes of a "
         "head that ist not tracked.");
}
//set cave hint
//
void VRCWProjectionVposFloor::setCaveHint() const
{
   ui.pwCaveHintLabel->setText("The viewer position and floor height are "
         "measured from the zero point in the direction of the coordinate "
         "axis.\n"
         "The viewer position is the position of the eyes of a "
         "head that is not tracked.");
}
