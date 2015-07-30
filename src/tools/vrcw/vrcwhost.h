#ifndef VRCWHOST_H
#define VRCWHOST_H

#include <QWidget>
#include "ui_vrcwhost.h"

#include <QStringListModel>
#include <QHostInfo>

#include "vrcwbase.h"
#include "vrcwutils.h"


class VRCWHost : public QWidget, public VRCWBase
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWHost(QWidget* parent = 0);
   ~VRCWHost();


   /*****
    * functions
    *****/
   //Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
   //Bearbeitung und Ueberpruefung der Eingaben im GUI
   int processGuiInput(const int& index, const QList<VRCWBase*>& vrcwList);

   //Liste der projection von VRCWProjectionHW uebernehmen und daraus die
   //minimale Anzahl an hosts bestimmen
   void setCalcNumHosts(const QStringList& projection);


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWHostClass ui;


   /*****
    * functions
    *****/
   //Auslesen des GUI
   opSys getGuiOs() const;
   execMode getGuiExec() const;

   //read hosts from input line (lineEdit)
   QVector<QString> readHostLineEdit() const;

   //OS-Version des Programms erfragen
   opSys getNativeOs() const;

   //GuiOs  auf nativeOs setzen, checkBox enable/disable
   void setGuiOs() const;

   //enable/disable Execution Mode abhaengig der definierten hosts
   void setEnableExecMode(const int& size);

   /*****
    * variables
    *****/
   QStringList hosts;
   QStringListModel* hostsModel;
   QStringList::size_type calcNumHosts;
   opSys natOs;
   bool expertMode;


private slots:
   void os_exec() const;
   void add();
   void remove();

   //Setzen der Variable expertMode und guiOs setzen
   void setExpertMode(const bool& changed);

};

#endif // VRCWHOST_H
