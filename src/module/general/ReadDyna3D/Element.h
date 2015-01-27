/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ELEMENT_H_
#define _ELEMENT_H_
/**************************************************************************\ 
 **                                                           (C)1999 RUS  **
 **                                                                        **
 ** Description:  Class-Declaration of object "Element"                    **
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

#include <util/coviseCompat.h>

enum Visibility
{
    VIS,
    INVIS
};

/*
//===========================================================================
// nodalNumbering
//===========================================================================

class NodalNumbering {

private:

  int nodeNo;   // node number (of LS-DYNA plot file)
  int coCon;    // entry of COVISE connectivity

public:

// constructors and destructors
NodalNumbering(){
nodeNo = -1;
coCon = -1;
};
~NodalNumbering() {} ;

//sets and gets
void set_nodeNo(int num);
void set_coCon(int entry);

};
*/

//===========================================================================
// Element
//===========================================================================

class Element
{

private:
    int coType; // COVISE element type
    int coElem; // COVISE element number
    int matNo; // material number (of LS-DYNA plot file)
    //int node[8];        // node numbers (of LS-DYNA plot file) determining the element
    Visibility visible; // element visible?

public:
    // constructors and destructors
    Element();
    ~Element(){};

    //sets and gets
    void set_coType(int type);
    void set_coElem(int num);
    void set_matNo(int num);
    //void set_node(int i, int num);
    void set_visible(Visibility vis);

    int get_coType()
    {
        return coType;
    };
    int get_coElem()
    {
        return coElem;
    };
    Visibility get_visible()
    {
        return visible;
    };
};
#endif // _ELEMENT_H_
