#ifndef VRCWSTART_H
#define VRCWSTART_H

#include <QWidget>
#include "ui_vrcwstart.h"

#include "vrcwbase.h"


class VRCWStart : public QWidget, public VRCWBase
{
   Q_OBJECT

public:
   /*****
   *constructor - destructor
   *****/
   VRCWStart(QWidget *parent = 0);
   ~VRCWStart();


   /*****
   *functions
   *****/
   // Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
   // Bearbeitung und Ueberpruefung der Eingaben im GUI
   int processGuiInput(const QList<VRCWBase*>& vrcwList);

   //Auslesen des GUI
   //Auslesen des Namens des Projektes (aus dem der config-Name erzeugt wird)
   QString getGuiProjName() const;

   //Setzen des GUI



   /*****
   *variables
   *****/


private:
   /*****
   *GUI Elements
   *****/
   Ui::VRCWStartClass ui;


   /*****
   *functions
   *****/

   // setup GUI
   void setupGui();
   void setupGuiExpert();
   void setGuiCoDirFalse() const;
   void setGuiCoDirTrue() const;

   QString getEnvVariable(const QString& key) const;
   bool testWritePerm(const QString& dir);


   /*****
   *variables
   *****/
   QString coDir;
   bool coDirSet;
   QString coConfDir;
   QString coConfName;
   QString coConfTmpDir;
   bool coConfDirWritable;

   bool expertMode;


private slots:
   void lineEdit_exec(const QString& fName);
   void chooseDir_exec();

   //Setzen der Variable expertMode
   void setExpertMode(const bool& changed);

};

#endif // VRCWSTART_H
