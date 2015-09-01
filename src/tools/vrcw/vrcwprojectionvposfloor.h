#ifndef VRCWPROJECTIONVPOSFLOOR_H
#define VRCWPROJECTIONVPOSFLOOR_H

#include <QWidget>
#include "ui_vrcwprojectionvposfloor.h"



class VRCWProjectionVposFloor : public QWidget
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
    VRCWProjectionVposFloor(QWidget *parent = 0);
    ~VRCWProjectionVposFloor();


    /*****
     * functions
     *****/
    //Auslesen des GUI
    QVector<int> getGuiVPos() const;
    int getGuiFloor() const;

    //Setzen des GUI
    void setGuiVPos(const QVector<int>& guiVPos) const;
    void setGuiFloor(const int& guiFloor) const;

    //enable/disable powerwall or cave hint
    void showCaveHint(const bool& yes = true) const;


private:
    /*****
     * GUI Elements
     *****/
    Ui::VRCWProjectionVposFloorClass ui;


    /*****
     * functions
     *****/
    //set powerwall and cave hint
    void setPwHint() const;
    void setCaveHint() const;

};

#endif // VRCWPROJECTIONVPOSFLOOR_H
