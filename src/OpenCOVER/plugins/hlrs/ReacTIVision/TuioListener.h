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

#ifndef INCLUDED_TUIOLISTENER_H
#define INCLUDED_TUIOLISTENER_H

#include "TuioObject.h"
#include "TuioCursor.h"

class TuioListener
{

public:
    virtual ~TuioListener(){};

    virtual void addTuioObject(TuioObject *tuioObject) = 0;
    virtual void updateTuioObject(TuioObject *tuioObject) = 0;
    virtual void removeTuioObject(TuioObject *tuioObject) = 0;

    virtual void addTuioCursor(TuioCursor *tuioCursor) = 0;
    virtual void updateTuioCursor(TuioCursor *tuioCursor) = 0;
    virtual void removeTuioCursor(TuioCursor *tuioCursor) = 0;

    virtual void refresh(long timestamp) = 0;
};

#endif /* INCLUDED_TUIOLISTENER_H */
