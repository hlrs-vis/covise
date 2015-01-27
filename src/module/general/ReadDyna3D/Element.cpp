/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1999 RUS  **
 **                                                                        **
 ** Description: Constructors and Member-Functions for object "Element"    **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Reiner Beller                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:                                                                  **
\**************************************************************************/

#include "Element.h"

//===========================================================================
// Element
//===========================================================================
Element::Element()
{
    coType = -1; // -1 : flag for unused storage
    coElem = -1;
    matNo = -1;
    //for (int i=0 ; i<8; i++) node[i] = -1;
    visible = VIS;
}

void Element::set_coType(int type)
{
    coType = type;
}

void Element::set_coElem(int num)
{
    coElem = num;
}

void Element::set_matNo(int num)
{
    matNo = num;
}

/*
void Element::set_node(int i, int num)
{
  assert( i>=0 && i<8 );
  node[i] = num;
}
*/

void Element::set_visible(Visibility vis)
{
    visible = vis;
}
