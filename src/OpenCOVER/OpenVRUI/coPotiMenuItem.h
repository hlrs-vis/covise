/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_POTIMENUITEM_H
#define CO_POTIMENUITEM_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coRowMenuItem.h>
#include <OpenVRUI/coValuePoti.h>

#include <string>

/** This class provides a coMenuItem which consists of a coValuePoti and a coLabel.
 */
namespace vrui
{

class OPENVRUIEXPORT coPotiMenuItem
    : public coRowMenuItem,
      public coValuePotiActor,
      public coAction
{
public:
    coPotiMenuItem(const std::string &name, float min, float max, float defaultValue,
                   vruiCOIM *iManager = 0, const std::string &interfaceName = "");
    virtual ~coPotiMenuItem();

    void setValue(float v);
    float getValue() const;

    void setMax(float m);
    void setMin(float m);
    void setInteger(bool i);
    void setIncrement(float incr);

    float getMax() const;
    float getMin() const;
    bool isInteger() const;
    bool isDiscrete() const;

    int hit(vruiHit *hit);
    void miss();

    virtual void selected(bool);
    virtual void doActionPress();
    virtual void doSecondActionPress();

    virtual void potiValueChanged(float, float, coValuePoti *, int context = -1);
    virtual void potiPressed(coValuePoti *, int context = -1);
    virtual void potiReleased(coValuePoti *, int context = -1);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    coValuePoti *poti; ///< actual poti interactor
};
}
#endif
