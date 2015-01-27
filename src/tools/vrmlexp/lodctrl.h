/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: lodctrl.cpp

	DESCRIPTION: A level of detail controller

	CREATED BY: Rolf Berteig

	HISTORY: 4/12/97

 *>	Copyright (c) 1997, All Rights Reserved.
 **********************************************************************/
#ifndef LOD_CTRL_H
#define LOD_CTRL_H

#define LOD_CONTROL_CLASS_ID Class_ID(0xbbe961a8, 0xa0ee7b7f)
#define LOD_CONTROL_CNAME GetString(IDS_RB_LODCONTROL)

#define LOD_UTILITY_CLASS_ID Class_ID(0x100d37ef, 0x1aa0ab84)
#define LOD_UTILITY_CNAME GetString(IDS_RB_LODUTILITU)
class LODCtrl : public StdControl
{
public:
    float min, max, bmin, bmax;
    WORD grpID;
    int order;
    BOOL viewport, highest;
};

#endif
