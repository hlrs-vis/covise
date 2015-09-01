#ifndef VRCWHANDSENSOR_H
#define VRCWHANDSENSOR_H

#include <QWidget>
#include "ui_vrcwhsensbindex.h"

#include "vrcwutils.h"


class VRCWHSensBIndex: public QWidget
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWHSensBIndex(QWidget *parent = 0);
   ~VRCWHSensBIndex();


   /*****
    * functions
    *****/
   //Auslesen des GUI
   QString getGuiHSensLabel() const;
   btnSys getGuiButtonSystem() const;
   QString getGuiButtonDevice() const;
   btnDrv getGuiButtonDriver() const;
   int getGuiBodyIndex() const;
   int getGuiVrcButtonAddr() const;

   //Setzen des GUI
   void setHSensLabel(const QString& hsLabel);
   void setButtonSystem(const btnSys& bSys);
   void setButtonDevice(const QString& bDev);
   void setButtonDriver(const btnDrv& bDrv);
   void setBodyIndex(const int& index);
   void setVrcButtonAddr(const int& addr);

   //set variable osData (operating system)
   void setOS_GUI(const opSys& os);

   //show only body index and hide button sys and dev and constHeaderDesc
   void bodyIndexOnly() const;

   //show only const Header Description and hide button sys and dev
   //and bodyIndex
   void constHeaderDescOnly(const QString& desc) const;

   //Layout hand/head sensor label for head sensor
   void headSensLayout()const;

   /*****
    * variables
    *****/
   //from vrcwtrackinghw
   trackSys tSysData;
   bool tHVrcData;

   //operating system from vrcwtrackinghw
   opSys osData;



private:
    /*****
     * GUI Elements
     *****/
    Ui::VRCWHSensBIndexClass ui;


    /*****
     * functions
     *****/
    void hideConstHeaderDesc() const;
    void setBtnDevComboBoxContent() const;
    void setBtnDrvComboBoxContent() const;



private slots:
   void trackSysChanged_exec(const trackSys& tSys);
   void trackHandleVrcChecked_exec(const bool& tHVrc);
   void buttonSys_exec() const;
   void buttonDev_exec(const QString& btDev) const;

};

#endif // VRCWHANDSENSOR_H
