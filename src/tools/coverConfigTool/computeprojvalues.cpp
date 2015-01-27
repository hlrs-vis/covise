/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** computeprojvalues.cpp
 ** 2004-01-20, Matthias Feurer
 ****************************************************************************/

#include "computeprojvalues.h"
#include <qstring.h>
#include <qmap.h>
//#include <iostream>
#include "covise.h"

ComputeProjValues::ComputeProjValues()
{
    newWidth = 0;
    newHeight = 0;
    newOrigin[0] = 0.0;
    newOrigin[1] = 0.0;
    newOrigin[2] = 0.0;
}

/*------------------------------------------------------------------------------
 ** computeNewProjDimensions():
 **   Computes the width, height and origin of the new proejection area
 **   if possible.
 **
 **   Parameters:
 **     projMap:           the projMap which stores the existing projection areas
 **                        and their values
 **     projComboboxText:  the current text of the projCombobox
 **     projTypeText:      the text of the newProjTypeCombobox
 **     whichSideItem:     the number of the item selected in the
 **                        whichSideCombobox. Possible values are left(0),
 **                        right(1), above(2), under(3), opposite(4).
 **
-------------------------------------------------------------------------------*/
void ComputeProjValues::computeProjDimensions(ProjectionAreaMap *projMap,
                                              QString projComboboxText,
                                              QString projTypeText,
                                              int whichSideItem)
{
    // find the item in the projMap
    if (projMap->find(projComboboxText) != projMap->end())
    {
        ProjectionArea p = (*projMap)[projComboboxText];
        ProjectionAreaMap::Iterator it = projMap->begin();

        QString valueString;
        switch (whichSideItem)
        {
        case 0: // left
            switch (p.getType())
            {
            case FRONT: // left of front -> front, left
                if (projTypeText == "FRONT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX() - p.getWidth();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                    // handle overlap in extra slot
                }
                else if (projTypeText == "LEFT")
                {
                    // if a corresponding right area exists, we assume that
                    // the left area has got the same width and height.
                    // So look if a right projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == RIGHT)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea right = it.data();
                        newWidth = right.getWidth();
                        newHeight = right.getHeight();

                        newOrigin[0] = p.getOriginX() - p.getWidth() / 2;
                        newOrigin[1] = right.getOriginY();
                        newOrigin[2] = right.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() - p.getWidth() / 2;
                        newOrigin[1] = p.getOriginY() - newWidth / 2;
                        newOrigin[2] = p.getOriginZ();
                    }
                }
                break;
            case BACK: // left of back -> back, right
                if (projTypeText == "BACK")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX() + p.getWidth();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "RIGHT")
                {
                    // if a corresponding left area exists, we assume that
                    // the right area has got the same width and height.
                    // So look if a left projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == LEFT)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea left = it.data();
                        newWidth = left.getWidth();
                        newHeight = left.getHeight();

                        newOrigin[0] = p.getOriginX() + p.getWidth() / 2;
                        newOrigin[1] = left.getOriginY();
                        newOrigin[2] = left.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() + p.getWidth() / 2;
                        newOrigin[1] = p.getOriginY() + newWidth / 2;
                        newOrigin[2] = p.getOriginZ();
                    }
                }
                break;
            case LEFT: // left of left -> left, back
                if (projTypeText == "LEFT")
                {
                    cout << "left of left" << endl;
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY() - p.getWidth();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "BACK")
                {
                    // if a corresponding front area exists, we assume that
                    // the back area has got the same width and height.
                    // So look if a front projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == FRONT)
                            found = true;
                        else
                            it++;
                    }

                    if (found == true)
                    {
                        ProjectionArea front = it.data();
                        cout << "left of left -> back, front found: " << it.data().getName()
                             << endl;
                        cout << "width: " << front.getWidth()
                             << endl;

                        newWidth = front.getWidth();
                        newHeight = front.getHeight();

                        newOrigin[0] = front.getOriginX();
                        newOrigin[1] = p.getOriginY() - p.getWidth() / 2;
                        newOrigin[2] = front.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() + newWidth / 2;
                        newOrigin[1] = p.getOriginY() - p.getWidth() / 2;
                        newOrigin[2] = p.getOriginZ();
                    }
                }
                break;
            case RIGHT: // left of right -> right, front
                if (projTypeText == "RIGHT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY() + p.getWidth();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "FRONT")
                {
                    // if a corresponding back area exists, we assume that
                    // the front area has got the same width and height.
                    // So look if a back projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == BACK)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea back = it.data();
                        newWidth = back.getWidth();
                        newHeight = back.getHeight();

                        newOrigin[0] = back.getOriginX();
                        newOrigin[1] = p.getOriginY() + p.getWidth() / 2;
                        newOrigin[2] = back.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() - newWidth / 2;
                        newOrigin[1] = p.getOriginY() + p.getWidth() / 2;
                        newOrigin[2] = p.getOriginZ();
                    }
                }
                break;
            case TOP: // left of top -> top, left
                if (projTypeText == "TOP")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX() - p.getWidth();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "LEFT")
                {
                    // if a corresponding right area exists, we assume that
                    // the left area has got the same width and height.
                    // So look if a right projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == RIGHT)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea right = it.data();
                        newWidth = right.getWidth();
                        newHeight = right.getHeight();

                        newOrigin[0] = p.getOriginX() - p.getWidth() / 2;
                        newOrigin[1] = right.getOriginY();
                        newOrigin[2] = right.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() - p.getWidth() / 2;
                        newOrigin[1] = p.getOriginY();
                        newOrigin[2] = p.getOriginZ() - newHeight / 2;
                    }
                }
                break;
            case BOTTOM: // left of bottom -> bottom, left
                if (projTypeText == "BOTTOM")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX() - p.getWidth();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "LEFT")
                {
                    // if a corresponding right area exists, we assume that
                    // the left area has got the same width and height.
                    // So look if a right projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == RIGHT)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea right = it.data();

                        newWidth = right.getWidth();
                        newHeight = right.getHeight();
                        newOrigin[0] = p.getOriginX() - p.getWidth() / 2;
                        newOrigin[1] = right.getOriginY();
                        newOrigin[2] = right.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() - p.getWidth() / 2;
                        newOrigin[1] = p.getOriginY();
                        newOrigin[2] = p.getOriginZ() + newHeight / 2;
                    }
                }
                break;
            }
            break;
        case 1: //right
            switch (p.getType())
            {
            case FRONT: // right of front -> front, right
                if (projTypeText == "FRONT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX() + p.getWidth();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "RIGHT")
                {
                    // Look if a left projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == LEFT)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea left = it.data();
                        newWidth = left.getWidth();
                        newHeight = left.getHeight();

                        newOrigin[0] = p.getOriginX() + p.getWidth() / 2;
                        newOrigin[1] = left.getOriginY();
                        newOrigin[2] = left.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() + p.getWidth() / 2;
                        newOrigin[1] = p.getOriginY() - newWidth / 2;
                        newOrigin[2] = p.getOriginZ();
                    }
                }
                break;
            case BACK: // right of back -> back, left
                if (projTypeText == "BACK")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX() - p.getWidth();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "LEFT")
                {
                    // Look if a right projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == RIGHT)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea right = it.data();
                        newWidth = right.getWidth();
                        newHeight = right.getHeight();

                        newOrigin[0] = p.getOriginX() - p.getWidth() / 2;
                        newOrigin[1] = right.getOriginY();
                        newOrigin[2] = right.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() + p.getWidth() / 2;
                        newOrigin[1] = p.getOriginY() + newWidth / 2;
                        newOrigin[2] = p.getOriginZ();
                    }
                }
                break;
            case LEFT: // right of left -> left, front
                if (projTypeText == "LEFT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY() + p.getWidth();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "FRONT")
                {
                    // Look if a back projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == BACK)
                            found = true;
                        else
                            it++;
                    }

                    if (found == true)
                    {
                        ProjectionArea back = it.data();
                        newWidth = back.getWidth();
                        newHeight = back.getHeight();

                        newOrigin[0] = back.getOriginX();
                        newOrigin[1] = p.getOriginY() + p.getWidth() / 2;
                        newOrigin[2] = back.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() + newWidth / 2;
                        newOrigin[1] = p.getOriginY() + p.getWidth() / 2;
                        newOrigin[2] = p.getOriginZ();
                    }
                }
                break;
            case RIGHT: // right of right -> right, back
                if (projTypeText == "RIGHT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY() - p.getWidth();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "BACK")
                {
                    // Look if a front projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == FRONT)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea front = it.data();
                        newWidth = front.getWidth();
                        newHeight = front.getHeight();

                        newOrigin[0] = front.getOriginX();
                        newOrigin[1] = p.getOriginY() - p.getWidth() / 2;
                        newOrigin[2] = front.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() - newWidth / 2;
                        newOrigin[1] = p.getOriginY() - p.getWidth() / 2;
                        newOrigin[2] = p.getOriginZ();
                    }
                }
                break;
            case TOP: // right of top -> top, right
                if (projTypeText == "TOP")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX() + p.getWidth();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "RIGHT")
                {
                    // Look if a left projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == LEFT)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea left = it.data();
                        newWidth = left.getWidth();
                        newHeight = left.getHeight();

                        newOrigin[0] = p.getOriginX() + p.getWidth() / 2;
                        newOrigin[1] = left.getOriginY();
                        newOrigin[2] = left.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() + p.getWidth() / 2;
                        newOrigin[1] = p.getOriginY();
                        newOrigin[2] = p.getOriginZ() - newHeight / 2;
                    }
                }
                break;
            case BOTTOM: // right of bottom -> bottom, right
                if (projTypeText == "BOTTOM")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX() + p.getWidth();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                }
                else if (projTypeText == "RIGHT")
                {
                    // Look if a left projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == LEFT)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea left = it.data();

                        newWidth = left.getWidth();
                        newHeight = left.getHeight();
                        newOrigin[0] = p.getOriginX() + p.getWidth() / 2;
                        newOrigin[1] = left.getOriginY();
                        newOrigin[2] = left.getOriginZ();
                    }
                    else
                    {
                        newWidth = p.getWidth();
                        newHeight = p.getHeight();
                        newOrigin[0] = p.getOriginX() + p.getWidth() / 2;
                        newOrigin[1] = p.getOriginY();
                        newOrigin[2] = p.getOriginZ() + newHeight / 2;
                    }
                }
                break;
            }
            break;
        case 2: // above
            switch (p.getType())
            {
            case FRONT: // above front -> front, top
                if (projTypeText == "FRONT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ() + p.getHeight();
                }
                else if (projTypeText == "TOP")
                {
                    // Look if a bottom projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == BOTTOM)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea bottom = it.data();
                        newWidth = bottom.getWidth();
                        newHeight = bottom.getHeight();

                        newOrigin[0] = bottom.getOriginX();
                        newOrigin[1] = bottom.getOriginY();
                        newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                    }
                    else
                    {
                        // look if a right or left area exists...
                        bool found2 = false;
                        while ((found2 == false) && (it != projMap->end()))
                        {
                            if ((it.data().getType() == LEFT) || (it.data().getType() == RIGHT))
                                found2 = true;
                            else
                                it++;
                        }
                        if (found2 == true)
                        {
                            ProjectionArea lr = it.data();
                            newWidth = p.getWidth();
                            // the height of the top area is the width
                            // of a left or right area
                            newHeight = lr.getWidth();

                            newOrigin[0] = p.getOriginX();
                            newOrigin[1] = lr.getOriginY();
                            newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                        }
                        else
                        {
                            // assume, the top area is square
                            newWidth = p.getWidth();
                            newHeight = p.getWidth();

                            newOrigin[0] = p.getOriginX();
                            newOrigin[1] = p.getOriginY() - newHeight / 2;
                            newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                        } // end if found2 == true

                    } // end if found1 == true
                } // end if TOP-area
                break;
            case BACK: // above back -> back, top
                if (projTypeText == "BACK")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ() + p.getHeight();
                }
                else if (projTypeText == "TOP")
                {
                    // Look if a bottom projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == BOTTOM)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea bottom = it.data();
                        newWidth = bottom.getWidth();
                        newHeight = bottom.getHeight();

                        newOrigin[0] = bottom.getOriginX();
                        newOrigin[1] = bottom.getOriginY();
                        newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                    }
                    else
                    {
                        // look if a right or left area exists...
                        bool found2 = false;
                        while ((found2 == false) && (it != projMap->end()))
                        {
                            if ((it.data().getType() == LEFT) || (it.data().getType() == RIGHT))
                                found2 = true;
                            else
                                it++;
                        }
                        if (found2 == true)
                        {
                            ProjectionArea lr = it.data();
                            newWidth = p.getWidth();
                            // the height of the top area is the width
                            // of a left or right area
                            newHeight = lr.getWidth();

                            newOrigin[0] = p.getOriginX();
                            newOrigin[1] = lr.getOriginY();
                            newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                        }
                        else
                        {
                            // assume, the top area is a square
                            newWidth = p.getWidth();
                            newHeight = p.getWidth();

                            newOrigin[0] = p.getOriginX();
                            newOrigin[1] = p.getOriginY() + newHeight / 2;
                            newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                        } // end if found2 == true

                    } // end if found1 == true
                } // end if TOP-area
                break;
            case LEFT: // above left -> left, top
                if (projTypeText == "LEFT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ() + p.getHeight();
                }
                else if (projTypeText == "TOP")
                {
                    // Look if a bottom projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == BOTTOM)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea bottom = it.data();
                        newWidth = bottom.getWidth();
                        newHeight = bottom.getHeight();

                        newOrigin[0] = bottom.getOriginX();
                        newOrigin[1] = bottom.getOriginY();
                        newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                    }
                    else
                    {
                        // look if a front or back area exists...
                        bool found2 = false;
                        while ((found2 == false) && (it != projMap->end()))
                        {
                            if ((it.data().getType() == FRONT) || (it.data().getType() == BACK))
                                found2 = true;
                            else
                                it++;
                        }
                        if (found2 == true)
                        {
                            ProjectionArea fb = it.data();
                            newHeight = p.getWidth();
                            newWidth = fb.getWidth();

                            newOrigin[0] = fb.getOriginX();
                            newOrigin[1] = p.getOriginY();
                            newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                        }
                        else
                        {
                            // assume, the top area is a square
                            newWidth = p.getWidth();
                            newHeight = p.getWidth();

                            newOrigin[0] = p.getOriginX() + newWidth / 2;
                            newOrigin[1] = p.getOriginY();
                            newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                        } // end if found2 == true

                    } // end if found1 == true
                } // end if TOP-area
                break;
            case RIGHT: // above right -> right, top
                if (projTypeText == "RIGHT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ() + p.getHeight();
                }
                else if (projTypeText == "TOP")
                {
                    // Look if a bottom projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == BOTTOM)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea bottom = it.data();
                        newWidth = bottom.getWidth();
                        newHeight = bottom.getHeight();

                        newOrigin[0] = bottom.getOriginX();
                        newOrigin[1] = bottom.getOriginY();
                        newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                    }
                    else
                    {
                        // look if a front or back area exists...
                        bool found2 = false;
                        while ((found2 == false) && (it != projMap->end()))
                        {
                            if ((it.data().getType() == FRONT) || (it.data().getType() == BACK))
                                found2 = true;
                            else
                                it++;
                        }
                        if (found2 == true)
                        {
                            ProjectionArea fb = it.data();
                            newHeight = p.getWidth();
                            newWidth = fb.getWidth();

                            newOrigin[0] = fb.getOriginX();
                            newOrigin[1] = p.getOriginY();
                            newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                        }
                        else
                        {
                            // assume, the top area is a square
                            newWidth = p.getWidth();
                            newHeight = p.getWidth();

                            newOrigin[0] = p.getOriginX() - newWidth / 2;
                            newOrigin[1] = p.getOriginY();
                            newOrigin[2] = p.getOriginZ() + p.getHeight() / 2;
                        } // end if found2 == true

                    } // end if found1 == true
                } // end if TOP-area
                break;
            case TOP:
                // not possible!!!
                break;
            case BOTTOM: // everything is possible.. but we don't allow that
                break;
            }
            break;
        case 3: // under
            switch (p.getType())
            {
            case FRONT: // under front -> front, bottom
                if (projTypeText == "FRONT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ() - p.getHeight();
                }
                else if (projTypeText == "BOTTOM")
                {
                    // Look if a top projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == TOP)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea top = it.data();
                        newWidth = top.getWidth();
                        newHeight = top.getHeight();

                        newOrigin[0] = top.getOriginX();
                        newOrigin[1] = top.getOriginY();
                        newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                    }
                    else
                    {
                        // look if a right or left area exists...
                        bool found2 = false;
                        while ((found2 == false) && (it != projMap->end()))
                        {
                            if ((it.data().getType() == LEFT) || (it.data().getType() == RIGHT))
                                found2 = true;
                            else
                                it++;
                        }
                        if (found2 == true)
                        {
                            ProjectionArea lr = it.data();
                            newWidth = p.getWidth();
                            // the height of the top area is the width
                            // of a left or right area
                            newHeight = lr.getWidth();

                            newOrigin[0] = p.getOriginX();
                            newOrigin[1] = lr.getOriginY();
                            newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                        }
                        else
                        {
                            // assume, the bottom area is square
                            newWidth = p.getWidth();
                            newHeight = p.getWidth();

                            newOrigin[0] = p.getOriginX();
                            newOrigin[1] = p.getOriginY() - newHeight / 2;
                            newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                        } // end if found2 == true

                    } // end if found1 == true
                } // end if TOP-area
                break;
            case BACK: // under back -> back, bottom
                if (projTypeText == "BACK")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ() - p.getHeight();
                }
                else if (projTypeText == "BOTTOM")
                {
                    // Look if a top projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == TOP)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea top = it.data();
                        newWidth = top.getWidth();
                        newHeight = top.getHeight();

                        newOrigin[0] = top.getOriginX();
                        newOrigin[1] = top.getOriginY();
                        newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                    }
                    else
                    {
                        // look if a right or left area exists...
                        bool found2 = false;
                        while ((found2 == false) && (it != projMap->end()))
                        {
                            if ((it.data().getType() == LEFT) || (it.data().getType() == RIGHT))
                                found2 = true;
                            else
                                it++;
                        }
                        if (found2 == true)
                        {
                            ProjectionArea lr = it.data();
                            newWidth = p.getWidth();
                            // the height of the top area is the width
                            // of a left or right area
                            newHeight = lr.getWidth();

                            newOrigin[0] = p.getOriginX();
                            newOrigin[1] = lr.getOriginY();
                            newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                        }
                        else
                        {
                            // assume, the bottom area is square
                            newWidth = p.getWidth();
                            newHeight = p.getWidth();

                            newOrigin[0] = p.getOriginX();
                            newOrigin[1] = p.getOriginY() + newHeight / 2;
                            newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                        } // end if found2 == true

                    } // end if found1 == true
                } // end if BACK-area
                break;
            case LEFT: // under left -> left, bottom
                if (projTypeText == "LEFT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ() - p.getHeight();
                }
                else if (projTypeText == "BOTTOM")
                {
                    // Look if a top projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == TOP)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea top = it.data();
                        newWidth = top.getWidth();
                        newHeight = top.getHeight();

                        newOrigin[0] = top.getOriginX();
                        newOrigin[1] = top.getOriginY();
                        newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                    }
                    else
                    {
                        // look if a front or back area exists...
                        bool found2 = false;
                        while ((found2 == false) && (it != projMap->end()))
                        {
                            if ((it.data().getType() == FRONT) || (it.data().getType() == BACK))
                                found2 = true;
                            else
                                it++;
                        }
                        if (found2 == true)
                        {
                            ProjectionArea fb = it.data();
                            newHeight = p.getWidth();
                            newWidth = fb.getWidth();

                            newOrigin[0] = fb.getOriginX();
                            newOrigin[1] = p.getOriginY();
                            newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                        }
                        else
                        {
                            // assume, the bottom area is a square
                            newWidth = p.getWidth();
                            newHeight = p.getWidth();

                            newOrigin[0] = p.getOriginX() + newWidth / 2;
                            newOrigin[1] = p.getOriginY();
                            newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                        } // end if found2 == true

                    } // end if found1 == true
                } // end if LEFT-area
                break;
            case RIGHT: // under right -> right, bottom
                if (projTypeText == "RIGHT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();
                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ() - p.getHeight();
                }
                else if (projTypeText == "BOTTOM")
                {
                    // Look if a top projection area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if (it.data().getType() == TOP)
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea top = it.data();
                        newWidth = top.getWidth();
                        newHeight = top.getHeight();

                        newOrigin[0] = top.getOriginX();
                        newOrigin[1] = top.getOriginY();
                        newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                    }
                    else
                    {
                        // look if a front or back area exists...
                        bool found2 = false;
                        while ((found2 == false) && (it != projMap->end()))
                        {
                            if ((it.data().getType() == FRONT) || (it.data().getType() == BACK))
                                found2 = true;
                            else
                                it++;
                        }
                        if (found2 == true)
                        {
                            ProjectionArea fb = it.data();
                            newHeight = p.getWidth();
                            newWidth = fb.getWidth();

                            newOrigin[0] = fb.getOriginX();
                            newOrigin[1] = p.getOriginY();
                            newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                        }
                        else
                        {
                            // assume, the bottom area is a square
                            newWidth = p.getWidth();
                            newHeight = p.getWidth();

                            newOrigin[0] = p.getOriginX() - newWidth / 2;
                            newOrigin[1] = p.getOriginY();
                            newOrigin[2] = p.getOriginZ() - p.getHeight() / 2;
                        } // end if found2 == true

                    } // end if found1 == true
                } // end if RIGHT-area
                break;
            case TOP: // under top -> all!
                break;
            case BOTTOM: // under bottom -> not possible!
                break;
            }
            break;
        case 4: // opposite
            switch (p.getType())
            {
            case FRONT: // opposite of front -> BACK
                cout << "opposite of front" << endl;
                if (projTypeText == "BACK")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();

                    newOrigin[0] = p.getOriginX();
                    newOrigin[2] = p.getOriginZ();
                    // look if a right or left area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if ((it.data().getType() == LEFT) || (it.data().getType() == RIGHT))
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea lr = it.data();
                        newOrigin[1] = lr.getOriginY() - lr.getWidth() / 2;
                    }
                    else
                    {
                        newOrigin[1] = -p.getOriginY();
                    }
                }
                break;
            case BACK: // opposite of back -> front
                if (projTypeText == "FRONT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();

                    newOrigin[0] = p.getOriginX();
                    newOrigin[2] = p.getOriginZ();
                    // look if a right or left area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if ((it.data().getType() == LEFT) || (it.data().getType() == RIGHT))
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea lr = it.data();
                        newOrigin[1] = lr.getOriginY() + lr.getWidth() / 2;
                    }
                    else
                    {
                        newOrigin[1] = -p.getOriginY();
                    }
                }
                break;
            case LEFT: // opposite of left -> right
                if (projTypeText == "RIGHT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();

                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                    // look if a top or bottom area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if ((it.data().getType() == TOP) || (it.data().getType() == BOTTOM) || (it.data().getType() == FRONT) || (it.data().getType() == BACK))
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea tbfb = it.data();
                        newOrigin[0] = tbfb.getOriginX() + tbfb.getWidth() / 2;
                    }
                    else
                    {
                        newOrigin[0] = -p.getOriginX();
                    }
                }
                break;
            case RIGHT: // opposite of right -> left
                if (projTypeText == "LEFT")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();

                    newOrigin[1] = p.getOriginY();
                    newOrigin[2] = p.getOriginZ();
                    // look if a top or bottom area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if ((it.data().getType() == TOP) || (it.data().getType() == BOTTOM) || (it.data().getType() == FRONT) || (it.data().getType() == BACK))
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea tbfb = it.data();
                        newOrigin[0] = tbfb.getOriginX() - tbfb.getWidth() / 2;
                    }
                    else
                    {
                        newOrigin[0] = -p.getOriginX();
                    }
                }
                break;
            case TOP: // opposite of top -> bottom
                if (projTypeText == "BOTTOM")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();

                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    // look if a front, back, left or right area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if ((it.data().getType() == FRONT) || (it.data().getType() == BACK) || (it.data().getType() == LEFT) || (it.data().getType() == RIGHT))
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea fblr = it.data();
                        newOrigin[2] = fblr.getOriginZ() - fblr.getHeight() / 2;
                    }
                    else
                    {
                        newOrigin[2] = -p.getOriginZ();
                    }
                }
                break;
            case BOTTOM: // opposite of bottom -> top
                if (projTypeText == "TOP")
                {
                    newWidth = p.getWidth();
                    newHeight = p.getHeight();

                    newOrigin[0] = p.getOriginX();
                    newOrigin[1] = p.getOriginY();
                    // look if a front, back, left or right area exists...
                    bool found = false;
                    while ((found == false) && (it != projMap->end()))
                    {
                        if ((it.data().getType() == FRONT) || (it.data().getType() == BACK) || (it.data().getType() == LEFT) || (it.data().getType() == RIGHT))
                            found = true;
                        else
                            it++;
                    }
                    if (found == true)
                    {
                        ProjectionArea fblr = it.data();
                        newOrigin[2] = fblr.getOriginZ() + fblr.getHeight() / 2;
                    }
                    else
                    {
                        newOrigin[2] = -p.getOriginZ();
                    }
                }
                break;
            }
            break;
        } // end switch
    } // end if (projMap->find ...
    else
    {
        cout << "projection area not found in projMap!" << endl;
    }
}

/*------------------------------------------------------------------------------
 ** computeNewProjOverlap():
 **   Computes the origin of the new projection area using the overlap
 **   value.
 **
 **   Parameters:
 **     p:                 the projection area to which we relate the new
 **                        projection area.
 **     originX, originY,
 **     originZ:           x,y, and z value of the origin of the new
 **                        projection area. These values are changed by the
 **                        function.
 **     projTypeText:      the text of the newProjTypeCombobox
 **     whichSideItem:     the number of the item selected in the
 **                        whichSideCombobox. Possible values are left(0),
 **                        right(1), above(2), under(3), opposite(4).
 **     overlapTermW:      overlap in mm, computed of the width of p
 **     overlapTermH:      overlap in mm, computed of the height of p
 **
-------------------------------------------------------------------------------*/
void ComputeProjValues::computeProjOverlap(ProjectionArea p,
                                           double *originX, double *originY, double *originZ,
                                           QString projTypeText,
                                           int whichSideItem,
                                           int overlapTermW,
                                           int overlapTermH)
{
    switch (whichSideItem)
    {
    case 0: // left
        switch (p.getType())
        {
        case FRONT: // left of front -> left, front
            if (projTypeText == "FRONT")
            {
                (*originX) += overlapTermW;
            }
            else if (projTypeText == "LEFT")
            {
                (*originY) += overlapTermW;
            }
            break;
        case BACK: // left of back -> back, right
            if (projTypeText == "BACK")
            {
                (*originX) -= overlapTermW;
            }
            else if (projTypeText == "RIGHT")
            {
                (*originY) -= overlapTermW;
            }
            break;
        case LEFT: // left of left -> left, back
            if (projTypeText == "LEFT")
            {
                (*originY) += overlapTermW;
            }
            else if (projTypeText == "BACK")
            {
                (*originX) -= overlapTermW;
            }
            break;
        case RIGHT: // left of right -> right, front
            if (projTypeText == "RIGHT")
            {
                (*originY) -= overlapTermW;
            }
            else if (projTypeText == "FRONT")
            {
                (*originX) += overlapTermW;
            }
            break;
        case TOP: // left of top -> top, left
            if (projTypeText == "TOP")
            {
                (*originX) += overlapTermW;
            }
            else if (projTypeText == "LEFT")
            {
                (*originZ) += overlapTermW;
            }
            break;
        case BOTTOM: // left of bottom -> bottom, left
            if (projTypeText == "BOTTOM")
            {
                (*originX) += overlapTermW;
            }
            else if (projTypeText == "LEFT")
            {
                (*originZ) -= overlapTermW;
            }
            break;
        }
        break;

    case 1: // right
        switch (p.getType())
        {
        case FRONT: // right of front -> front, right
            if (projTypeText == "FRONT")
            {
                (*originX) -= overlapTermW;
            }
            else if (projTypeText == "RIGHT")
            {
                (*originY) += overlapTermW;
            }
            break;
        case BACK: // right of back -> back, left
            if (projTypeText == "BACK")
            {
                (*originX) += overlapTermW;
            }
            else if (projTypeText == "LEFT")
            {
                (*originY) -= overlapTermW;
            }
            break;
        case LEFT: // right of left -> left, front
            if (projTypeText == "LEFT")
            {
                (*originY) -= overlapTermW;
            }
            else if (projTypeText == "FRONT")
            {
                (*originX) -= overlapTermW;
            }
            break;
        case RIGHT: // right of right -> right, back
            if (projTypeText == "RIGHT")
            {
                (*originY) += overlapTermW;
            }
            else if (projTypeText == "BACK")
            {
                (*originX) += overlapTermW;
            }
            break;
        case TOP: // right of top -> top, right
            if (projTypeText == "TOP")
            {
                (*originX) -= overlapTermW;
            }
            else if (projTypeText == "RIGHT")
            {
                (*originZ) += overlapTermW;
            }
            break;
        case BOTTOM: // right of bottom -> bottom, right
            if (projTypeText == "BOTTOM")
            {
                (*originX) -= overlapTermW;
            }
            else if (projTypeText == "right")
            {
                (*originZ) -= overlapTermW;
            }
            break;
        }
        break;

    case 2: // above
        switch (p.getType())
        {
        case FRONT: // above of front -> front, top
            if (projTypeText == "FRONT")
            {
                (*originZ) -= overlapTermH;
            }
            else if (projTypeText == "TOP")
            {
                (*originY) += overlapTermH;
            }
            break;
        case BACK: // above back -> back, top
            if (projTypeText == "BACK")
            {
                (*originZ) -= overlapTermH;
            }
            else if (projTypeText == "TOP")
            {
                (*originY) -= overlapTermH;
            }
            break;
        case LEFT: // above left -> left, top
            if (projTypeText == "LEFT")
            {
                (*originZ) -= overlapTermH;
            }
            else if (projTypeText == "top")
            {
                (*originX) -= overlapTermH;
            }
            break;
        case RIGHT: // above right -> right, top
            if (projTypeText == "RIGHT")
            {
                (*originY) -= overlapTermH;
            }
            else if (projTypeText == "TOP")
            {
                (*originX) += overlapTermH;
            }
            break;
        case TOP: // above top -> impossible.
            break;
        case BOTTOM: // above bottom -> everything,  but we don't allow that.
            break;
        }
        break;

    case 3: // under
        switch (p.getType())
        {
        case FRONT: // under front -> front, bottom
            if (projTypeText == "FRONT")
            {
                (*originZ) += overlapTermH;
            }
            else if (projTypeText == "BOTTOM")
            {
                (*originY) += overlapTermH;
            }
            break;
        case BACK: // under back -> back, bottom
            if (projTypeText == "BACK")
            {
                (*originZ) += overlapTermH;
            }
            else if (projTypeText == "BOTTOM")
            {
                (*originY) -= overlapTermH;
            }
            break;
        case LEFT: // under left -> left, bottom
            if (projTypeText == "LEFT")
            {
                (*originZ) += overlapTermH;
            }
            else if (projTypeText == "bottom")
            {
                (*originX) -= overlapTermH;
            }
            break;
        case RIGHT: // under right -> right, bottom
            if (projTypeText == "RIGHT")
            {
                (*originZ) += overlapTermH;
            }
            else if (projTypeText == "BOTTOM")
            {
                (*originX) += overlapTermH;
            }
            break;
        case TOP: // under top -> everything, but we don't allow that.
            break;
        case BOTTOM: // under bottom -> impossible.
            break;
        }
        break;

    case 4: // opposite
        // projection areas which are on opposite sides cannot overlap!
        break;
    }
}

/*------------------------------------------------------------------------------
 ** computeNewOrigin:
 **   Computes the new origin of the new projection area using the changed
 **   width and height.
 **   value.
 **
 **   Parameters:
 **     p:                 the projection area to which we relate the new
 **                        projection area.
 **     originX, originY,
 **     originZ:           x,y, and z value of the origin of the new
 **                        projection area. These values are changed by the
 **                        function.
 **     projTypeText:      the text of the newProjTypeCombobox
 **     whichSideItem:     the number of the item selected in the
 **                        whichSideCombobox. Possible values are left(0),
 **                        right(1), above(2), under(3), opposite(4).
 **     diffWidth:         = newWidth - oldWidth
 **     diffHeight:        = newHeight - oldHeight
 **
-------------------------------------------------------------------------------*/
void ComputeProjValues::computeNewOrigin(ProjectionArea p,
                                         double *originX,
                                         double *originY,
                                         double *originZ,
                                         QString projTypeText,
                                         int whichSideItem,
                                         double diffWidth,
                                         double diffHeight)
{
    switch (whichSideItem)
    {
    case 0: // left
        switch (p.getType())
        {
        case FRONT: // left of front -> front, left
            if (projTypeText == "FRONT")
            {
                (*originX) -= diffWidth / 2;
            }
            else if (projTypeText == "LEFT")
            {
                (*originY) -= diffWidth / 2;
            }
            break;
        case BACK: // left of back -> back, right
            if (projTypeText == "BACK")
            {
                (*originX) += diffWidth / 2;
            }
            else if (projTypeText == "RIGHT")
            {
                (*originY) += diffWidth / 2;
            }
            break;
        case LEFT: // left of left -> left, back
            if (projTypeText == "LEFT")
            {
                (*originY) -= diffWidth / 2;
            }
            else if (projTypeText == "BACK")
            {
                (*originX) += diffWidth / 2;
            }
            break;
        case RIGHT: // left of right -> right, front
            if (projTypeText == "RIGHT")
            {
                (*originY) += diffWidth / 2;
            }
            else if (projTypeText == "FRONT")
            {
                (*originX) -= diffWidth / 2;
            }
            break;
        case TOP: // left of top -> top, left
            if (projTypeText == "TOP")
            {
                (*originX) -= diffWidth / 2;
            }
            else if (projTypeText == "LEFT")
            {
                (*originZ) -= diffHeight / 2;
            }
            break;
        case BOTTOM: // left of bottom -> bottom, left
            if (projTypeText == "BOTTOM")
            {
                (*originX) -= diffWidth / 2;
            }
            else if (projTypeText == "LEFT")
            {
                (*originZ) += diffHeight / 2;
            }
            break;
        }
        break;
    case 1: // right
        switch (p.getType())
        {
        case FRONT: // right of front -> front, right
            if (projTypeText == "FRONT")
            {
                (*originX) += diffWidth / 2;
            }
            else if (projTypeText == "RIGHT")
            {
                (*originY) -= diffWidth / 2;
            }
            break;
        case BACK: // right of back -> back, left
            if (projTypeText == "BACK")
            {
                (*originX) -= diffWidth / 2;
            }
            else if (projTypeText == "LEFT")
            {
                (*originY) += diffWidth / 2;
            }
            break;
        case LEFT: // right of left -> left, front
            if (projTypeText == "LEFT")
            {
                (*originY) += diffWidth / 2;
            }
            else if (projTypeText == "FRONT")
            {
                (*originX) += diffWidth / 2;
            }
            break;
        case RIGHT: // right of right -> right, back
            if (projTypeText == "RIGHT")
            {
                (*originY) -= diffWidth / 2;
            }
            else if (projTypeText == "BACK")
            {
                (*originX) -= diffWidth / 2;
            }
            break;
        case TOP: // right of top -> top, right
            if (projTypeText == "TOP")
            {
                (*originX) += diffWidth / 2;
            }
            else if (projTypeText == "RIGHT")
            {
                (*originZ) -= diffHeight / 2;
            }
            break;
        case BOTTOM: // right of bottom -> bottom, right
            if (projTypeText == "BOTTOM")
            {
                (*originX) += diffWidth / 2;
            }
            else if (projTypeText == "RIGHT")
            {
                (*originZ) += diffHeight / 2;
            }
            break;
        }
        break;
    case 2: // above
        switch (p.getType())
        {
        case FRONT: // above of front -> front, top
            if (projTypeText == "FRONT")
            {
                (*originZ) += diffHeight / 2;
            }
            else if (projTypeText == "TOP")
            {
                (*originY) -= diffHeight / 2;
            }
            break;
        case BACK: // above back -> back, top
            if (projTypeText == "BACK")
            {
                (*originZ) += diffHeight / 2;
            }
            else if (projTypeText == "TOP")
            {
                (*originY) += diffHeight / 2;
            }
            break;
        case LEFT: // above left -> left, top
            if (projTypeText == "LEFT")
            {
                (*originZ) += diffHeight / 2;
            }
            else if (projTypeText == "TOP")
            {
                (*originX) += diffWidth / 2;
            }
            break;
        case RIGHT: // above right -> right, top
            if (projTypeText == "RIGHT")
            {
                (*originY) += diffHeight / 2;
            }
            else if (projTypeText == "TOP")
            {
                (*originX) -= diffWidth / 2;
            }
            break;
        case TOP: // above top -> impossible.
            break;
        case BOTTOM: // above bottom -> everything,  but we don't allow that.
            break;
        }
        break;
    case 3: // under
        switch (p.getType())
        {
        case FRONT: // under front -> front, bottom
            if (projTypeText == "FRONT")
            {
                (*originZ) -= diffHeight / 2;
            }
            else if (projTypeText == "BOTTOM")
            {
                (*originY) -= diffHeight / 2;
            }
            break;
        case BACK: // under back -> back, bottom
            if (projTypeText == "BACK")
            {
                (*originZ) -= diffHeight / 2;
            }
            else if (projTypeText == "BOTTOM")
            {
                (*originY) += diffHeight / 2;
            }
            break;
        case LEFT: // under left -> left, bottom
            if (projTypeText == "LEFT")
            {
                (*originZ) -= diffHeight / 2;
            }
            else if (projTypeText == "BOTTOM")
            {
                (*originX) += diffWidth / 2;
            }
            break;
        case RIGHT: // under right -> right, bottom
            if (projTypeText == "RIGHT")
            {
                (*originZ) -= diffHeight / 2;
            }
            else if (projTypeText == "BOTTOM")
            {
                (*originX) -= diffWidth / 2;
            }
            break;
        case TOP: // under top -> everything, but we don't allow that.
            break;
        case BOTTOM: // under bottom -> impossible.
            break;
        }
        break;
    case 4: // opposite
        // projection areas which are on opposite sides cannot overlap!
        break;
    }
}

int ComputeProjValues::getNewWidth()
{
    return newWidth;
}

int ComputeProjValues::getNewHeight()
{
    return newHeight;
}

double ComputeProjValues::getNewOriginX()
{
    return newOrigin[0];
}

double ComputeProjValues::getNewOriginY()
{
    return newOrigin[1];
}

double ComputeProjValues::getNewOriginZ()
{
    return newOrigin[2];
}
