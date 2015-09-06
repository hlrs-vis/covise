#ifndef VRCWSENSORTRACKSYSDIM_H_
#define VRCWSENSORTRACKSYSDIM_H_

#include <QWidget>
#include "ui_vrcwsensortracksysdim.h"

class sensTrackSysDim;


class VRCWSensorTrackSysDim: public QWidget
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWSensorTrackSysDim(QWidget *parent = 0);
   ~VRCWSensorTrackSysDim();


   /*****
    * functions
    *****/
   //Auslesen des GUI
   sensTrackSysDim* getGuiSensTrackSysDim() const;

   //Setzen des GUI
   void setSensTrackSysLabel(const QString& stsLabel);
   void setSensTrackSysDesc(const QString& stsDesc);
   void hideSensTrackSysDesc() const;
   void setSensTrackSysOffset(const sensTrackSysDim* stsd);
   void setSensTrackSysOrient(const sensTrackSysDim* stsd);



private:
    /*****
     * GUI Elements
     *****/
    Ui::VRCWSensorTrackSysDimClass ui;

};

#endif /* VRCWSENSORTRACKSYSDIM_H_ */
