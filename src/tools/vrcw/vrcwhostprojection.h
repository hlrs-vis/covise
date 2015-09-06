#ifndef VRCWHOSTPROJECTION_H
#define VRCWHOSTPROJECTION_H

#include <QWidget>
#include "ui_vrcwhostprojection.h"

#include "vrcwbase.h"

class ListStrListTModel;


class VRCWHostProjection: public QWidget, public VRCWBase
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWHostProjection(QWidget* parent = 0);
   ~VRCWHostProjection();


   /*****
    * functions
    *****/
   // Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
   // Bearbeitung und Ueberpruefung der Eingaben im GUI
   int processGuiInput(const QList<VRCWBase*>& vrcwList);

   //Entgegennehmen der ProjectionData
   void setGuiProjectionData(const QStringList& projection);

   //Entgegennehmen der HostData
   void setGuiHostData(const QStringList& host);


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWHostProjectionClass ui;


   /*****
    * functions
    *****/
   //Setzen der host-projection-Eintraege in der GUI
   void setGui();


   /*****
    * variables
    *****/
   //speichert die Daten in der Reihenfolge:
   //Index 0 = projection, 1 = eye, 2 = host
   //siehe Slot add, und Konstruktor: tableHeader
   ListStrListTModel* hostProjectionModel;
   QStringList tableHeader;
   //
   QStringList projectionData;
   QStringList hostData;
   //
   bool dataChanged;


   /*****
    * constants
    *****/
   const QString NOT_SEL;


private slots:
   void add() const;
   void remove();

};

#endif // VRCWHOSTPROJECTION_H
