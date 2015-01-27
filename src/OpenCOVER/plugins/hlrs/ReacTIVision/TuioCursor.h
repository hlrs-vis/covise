/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
	TUIO C++ Library - part of the reacTIVision project
	http://reactivision.sourceforge.net/

	Copyright (c) 2005-2008 Martin Kaltenbrunner <mkalten@iua.upf.edu>
	
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#ifndef INCLUDED_TUIOCURSOR_H
#define INCLUDED_TUIOCURSOR_H

#include <list>
#include <math.h>
#include "TuioContainer.h"

class TuioCursor : public TuioContainer
{

protected:
    int finger_id;
    std::list<TuioPoint> path;

public:
    TuioCursor(long s_id, int f_id, float xpos, float ypos)
        : TuioContainer(s_id, xpos, ypos)
    {
        this->finger_id = f_id;
    };

    TuioCursor(TuioCursor *tuioCursor)
        : TuioContainer(tuioCursor)
    {
        this->finger_id = tuioCursor->getFingerID();
    };

    ~TuioCursor(){};

    int getFingerID()
    {
        return finger_id;
    };
};

#endif
