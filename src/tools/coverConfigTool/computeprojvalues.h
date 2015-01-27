/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** computeprojvalues.h
 ** 2004-01-20, Matthias Feurer
 ****************************************************************************/

#ifndef COMPUTEPROJVALUES_H
#define COMPUTEPROJVALUES_H

#include <qstring.h>
#include <qmap.h>
#include "projectionarea.h"

class ComputeProjValues
{
public:
    ComputeProjValues();

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
    void computeProjDimensions(ProjectionAreaMap *projMap,
                               QString projComboboxText,
                               QString projTypeText,
                               int whichSideItem);

    /*------------------------------------------------------------------------------
       ** computeNewProjOverlap:
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
    void computeProjOverlap(ProjectionArea p,
                            double *originX, double *originY, double *originZ,
                            QString projTypeText,
                            int whichSideItem,
                            int overlapTermW,
                            int overlapTermH);

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
    void computeNewOrigin(ProjectionArea p,
                          double *originX, double *originY, double *originZ,
                          QString projTypeText, int whichSideItem,
                          double diffWidth, double diffHeight);

    int getNewWidth();
    int getNewHeight();
    double getNewOriginX();
    double getNewOriginY();
    double getNewOriginZ();

private:
    int newWidth;
    int newHeight;
    double newOrigin[3];
};
#endif // COMPUTEPROJVALUES_H
