/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NAME_H_
#define _NAME_H_

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// <<<<<<<<<<<<<<<<<<<<<<<<<<  Name.h  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// <<<<<<<<<<<<<<<<<<<<<<<<<<          >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<  by  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<      >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// <<<<<<<<<<<<<<<<<<<<  Bjoern Foldenauer   >>>>>>>>>>>>>>>>>>>>>>>>>>>>
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <LEDA/basic.h>
#include <LEDA/string.h>
#include <LEDA/list.h>

class Name
{
    int _pos_in_list;
    string _id;
    int _list_type;

public:
    Name(); // Konstr.
    Name(const Name &namecpy);
    ~Name();

    void fill(string id, int list_type, int pos_in_list);
    string getid() const;
    int getpos_in_list() const;
    int getlist_type() const;

    friend ostream &operator<<(ostream &ausgabe, const Name &namecpy);
    friend istream &operator>>(istream &eingabe, Name &namecpy);
};
#endif
