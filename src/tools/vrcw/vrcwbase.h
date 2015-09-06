#ifndef VRCWBASE_H_
#define VRCWBASE_H_

#include <QList>


class VRCWBase
{
public:
   /*****
    * constructor - destructor
    *****/
   VRCWBase();
   virtual ~VRCWBase();


   /*****
    * functions
    *****/
   // Bearbeitung und Ueberpruefung der Eingaben im GUI
   virtual int processGuiInput(const QList<VRCWBase*>& vrcwList);
   virtual int processGuiInput(const int& index,
         const QList<VRCWBase*>& vrcwList);

};

#endif /* VRCWBASE_H_ */
