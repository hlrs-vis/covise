#ifndef VRCWTEMPLATE_H
#define VRCWTEMPLATE_H

#include <QWidget>
#include "ui_vrcwtemplate.h"

#include "vrcwbase.h"


class VRCWTemplate : public QWidget, public VRCWBase
{
   Q_OBJECT

public:
   /*****
   *constructor - destructor
   *****/
   VRCWTemplate(QWidget *parent = 0);
   ~VRCWTemplate();


   /*****
   *functions
   *****/
   // Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
   // Bearbeitung und Ueberpruefung der Eingaben im GUI
   int processGuiInput();


   /*****
   *variables
   *****/


private:
   /*****
   *GUI Elements
   *****/
   Ui::VRCWTemplateClass ui;


   /*****
   *functions
   *****/


   /*****
   *variables
   *****/

};

#endif // VRCWTEMPLATE_H
