#ifndef VRCWTRACKINGHW_H
#define VRCWTRACKINGHW_H

#include <QWidget>
#include "ui_vrcwtrackinghw.h"

#include "vrcwbase.h"
#include "vrcwutils.h"

class VRCWHSensBIndex;
class VRCWPerson;


class VRCWTrackingHw : public QWidget, public VRCWBase
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWTrackingHw(QWidget* parent = 0);
   ~VRCWTrackingHw();


   /*****
    * functions
    *****/
   // Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
   // Bearbeitung und Ueberpruefung der Eingaben im GUI
   int processGuiInput(const int& index, const QList<VRCWBase*>& vrcwList);

   //set variable osData (operating system)
   //and execute setupGui()
   void setOS_GUI(const opSys& os);

private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWTrackingHwClass ui;


   /*****
    * functions
    *****/
   // Auslesen des GUI
   trackHandle getGuiTrackHandle() const;
   int getGuiVrcPort() const;
   trackSys getGuiTrackSys() const;
   bool getGuiCheckArtHost() const;
   QString getGuiArtHost() const;
   int getGuiArtHostSPort() const;
   int getGuiArtRPort() const;
   QString getGuiViconHost() const;
   QString getGuiPolFobSPort() const;
   QString getGuiMoStarIPAddr() const;
   int getGuiNumHandSens() const;
   int getGuiNumHeadSens() const;
   int getGuiNumPersons() const;

   //setup GUI depending on osData
   void setupGui() const;

   //Setzen der GUI
   void setPersonsHandSens(const int& numHS);
   void setPersonsHeadSens(const int& numHS);


   /*****
    * variables
    *****/
   VRCWHSensBIndex* handSensor_1;
   VRCWHSensBIndex* handSensor_2;
   VRCWHSensBIndex* handSensor_3;
   VRCWHSensBIndex* headSensor_0;
   VRCWHSensBIndex* headSensor_1;
   VRCWHSensBIndex* headSensor_2;
   VRCWHSensBIndex* headSensor_3;
   VRCWPerson* person_1;
   VRCWPerson* person_2;
   VRCWPerson* person_3;
   VRCWPerson* person_4;
   VRCWPerson* person_5;
   VRCWPerson* person_6;
   VRCWPerson* person_7;
   VRCWPerson* person_8;
   VRCWPerson* person_9;
   QVector<VRCWHSensBIndex*> handSensors;
   QVector<VRCWHSensBIndex*> headSensors;
   QVector<VRCWPerson*> persons;
   QVector<QWidget*> personWidgets;
   bool expertMode;

   //operating system from VRCWHost
   opSys osData;

private slots:
   void trackHandleParty_exec();
   void trackSys_exec();
   void numHandSens_exec(const int& numHS);
   void numHeadSens_exec(const int& numHS);
   void numPersons_exec(const int& numP) const;
   void artHostIP_Exec(const bool& checked) const;
   void polFobSPort_exec(const QString& pfSP) const;

   //Setzen der Variable expertMode
   void setExpertMode(const bool& changed);

signals:
   void trackSysChanged(const trackSys& tSys);
   void trackHandleVrcChecked(const bool& tHVrc);

};

#endif // VRCWTRACKINGHW_H
