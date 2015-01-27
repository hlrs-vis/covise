/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_NULL_BUTTON_H
#define SG_VRUI_NULL_BUTTON_H

#include <util/coTypes.h>

#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <OpenVRUI/coButtonGeometry.h>

class SGVruiTransformNode;

class SGVRUIEXPORT SGVruiNullButton : public vruiButtonProvider
{

public:
    SGVruiNullButton(coButtonGeometry *button);
    virtual ~SGVruiNullButton();

    virtual float getWidth() const
    {
        return 500.0f;
    }
    virtual float getHeight() const
    {
        return 500.0f;
    }

    virtual vruiTransformNode *getDCS();
    virtual void switchGeometry(coButtonGeometry::ActiveGeometry /*active*/)
    {
    }

    void createGeometry();

protected:
    virtual void resizeGeometry();

    SGVruiTransformNode *myDCS;
};
#endif
