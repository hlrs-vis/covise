#include "vrcwperson.h"

#include "datatypes.h"



/*****
 * constructor - destructor
 *****/

VRCWPerson::VRCWPerson(QWidget *parent) : QWidget(parent)
{
   ui.setupUi(this);
}

VRCWPerson::~VRCWPerson()
{

}


/*****
 * public functions
 *****/

// Auslesen des GUI
//
personVal* VRCWPerson::getGuiPerson() const
{
   personVal* pers = new personVal();

   pers->personLabel = ui.trackedPersonLabel->text();
   pers->handSens = ui.handSensorComboBox->currentText();
   pers->headSens = ui.headSensorComboBox->currentText();

   return pers;
}

//Setzen des GUI
//
void VRCWPerson::setPersonsLabel(const QString& pLabel)
{
   ui.trackedPersonLabel->setText(pLabel);
}

void VRCWPerson::setHandSensCBoxContent(const QStringList& hSens)
{
   ui.handSensorComboBox->clear();
   ui.handSensorComboBox->addItems(hSens);
}

void VRCWPerson::setHeadSensCBoxContent(const QStringList& hSens)
{
   ui.headSensorComboBox->clear();
   ui.headSensorComboBox->addItems(hSens);
}
