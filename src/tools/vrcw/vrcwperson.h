#ifndef VRCWPERSON_H_
#define VRCWPERSON_H_

#include <QWidget>
#include "ui_vrcwperson.h"

class personVal;


class VRCWPerson: public QWidget
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWPerson(QWidget *parent = 0);
   ~VRCWPerson();


   /*****
    * functions
    *****/
   //Auslesen des GUI
   personVal* getGuiPerson() const;

   //Setzen des GUI
   void setPersonsLabel(const QString& pLabel);
   void setHandSensCBoxContent(const QStringList& hSens);
   void setHeadSensCBoxContent(const QStringList& hSens);



private:
    /*****
     * GUI Elements
     *****/
    Ui::VRCWPersonClass ui;

};

#endif /* VRCWPERSON_H_ */
