/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream.h>
#include "vvsllist.h"

//----------------------------------------------------------------------------
/** Test program for vvSLList class.
 */
int main(int, char **)
{
    vvSLList<unsigned char *> *testList;
    vvSLList<unsigned char *> *newList;
    unsigned char data[] = { 0, 1, 2, 3, 4 };
    unsigned char newData[] = { 100, 101 };

    testList = new vvSLList<unsigned char *>;
    cerr << "Number of list elements: " << testList->count() << endl;

    cerr << "\nCreate list of 4 elements:" << endl;
    testList->append(&data[0], false);
    testList->append(&data[1], false);
    testList->insertAfter(&data[2], false);
    testList->insertBefore(&data[3], false);
    testList->print();

    cerr << "\nRemove item at index 2:" << endl;
    testList->makeCurrent(2);
    testList->remove();
    testList->print();

    cerr << "\nAppend an element:" << endl;
    testList->append(&data[4], false);
    testList->print();

    cerr << "\nFind element of value 1 and remove it:" << endl;
    if (testList->find(&data[1]))
        testList->remove();
    testList->print();
    cerr << "Current element is at index: " << testList->getIndex() << endl;

    cerr << "\nRemove element before:" << endl;
    testList->previous();
    testList->remove();
    testList->print();

    cerr << "\nMerge another list:" << endl;
    newList = new vvSLList<unsigned char *>;
    newList->append(&newData[0], false);
    newList->append(&newData[1], false);
    testList->merge(newList);
    testList->print();
    delete newList;

    cerr << "Number of list elements: " << testList->count() << endl;

    delete testList;
    return 0;
}
