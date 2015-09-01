#include "vrcwsensortracksysdim.h"

#include "datatypes.h"



/*****
 * constructor - destructor
 *****/

VRCWSensorTrackSysDim::VRCWSensorTrackSysDim(QWidget *parent) :
   QWidget(parent)
{
   ui.setupUi(this);
}

VRCWSensorTrackSysDim::~VRCWSensorTrackSysDim()
{

}


/*****
 * public functions
 *****/

// Auslesen des GUI
//
sensTrackSysDim* VRCWSensorTrackSysDim::getGuiSensTrackSysDim() const
{
   sensTrackSysDim* stsDim = new sensTrackSysDim();

   stsDim->sensLabel = ui.sensTrackSysLabel->text();
   stsDim->x = ui.OffsetXSpinBox->value();
   stsDim->y = ui.OffsetYSpinBox->value();
   stsDim->z = ui.OffsetZSpinBox->value();
   stsDim->h = ui.OrientationHSpinBox->value();
   stsDim->p = ui.OrientationPSpinBox->value();
   stsDim->r = ui.OrientationRSpinBox->value();

   return stsDim;
}

//Setzen des GUI
//
void VRCWSensorTrackSysDim::setSensTrackSysLabel(const QString& stsLabel)
{
   ui.sensTrackSysLabel->setText(stsLabel);
}

void VRCWSensorTrackSysDim::setSensTrackSysDesc(const QString& stsDesc)
{
   ui.descLabel->setText(stsDesc);
}

void VRCWSensorTrackSysDim::hideSensTrackSysDesc() const
{
   ui.descLabel->hide();
}

void VRCWSensorTrackSysDim::setSensTrackSysOffset(const sensTrackSysDim* stsd)
{
   ui.OffsetXSpinBox->setValue(stsd->x);
   ui.OffsetYSpinBox->setValue(stsd->y);
   ui.OffsetZSpinBox->setValue(stsd->z);
}

void VRCWSensorTrackSysDim::setSensTrackSysOrient(const sensTrackSysDim* stsd)
{
   ui.OrientationHSpinBox->setValue(stsd->h);
   ui.OrientationPSpinBox->setValue(stsd->p);
   ui.OrientationRSpinBox->setValue(stsd->r);
}
