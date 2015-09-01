#include "vrcwtemplate.h"


/*****
 * constructor - destructor
 *****/

VRCWTemplate::VRCWTemplate(QWidget *parent) :
   QWidget(parent)
{
   ui.setupUi(this);
}

VRCWTemplate::~VRCWTemplate()
{

}

/*****
 *public functions
 *****/

//Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
//Bearbeitung und Ueberpruefung der Eingaben im GUI
//
int VRCWTemplate::processGuiInput()
{
   return 9;
}

/*****
 *private functions
 *****/
