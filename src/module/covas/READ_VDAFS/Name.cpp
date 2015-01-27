/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Name.h"

Name::Name()
{
}

Name::~Name()
{
}

void Name::fill(string id, int list_type, int pos_in_list)
{
    _id = id;
    _list_type = list_type;
    _pos_in_list = pos_in_list;
}

int Name::getpos_in_list() const
{
    return _pos_in_list;
}

int Name::getlist_type() const
{
    return _list_type;
}

string Name::getid() const
{
    return _id;
}

Name::Name(const Name &namecpy) //Copy
{
    _pos_in_list = namecpy._pos_in_list;
    _id = namecpy._id;
    _list_type = namecpy._list_type;
}

ostream &operator<<(ostream &ausgabe, const Name &name)
{
    ausgabe << "<" << name._pos_in_list << "," << name._id
            << "," << name._list_type << ">";
    return ausgabe;
}

istream &operator>>(istream &eingabe, Name &name)
{

    int c;

    if ((c = eingabe.get()) != '<')
    {
        eingabe.clear(ios::badbit | eingabe.rdstate());
        return eingabe;
    }

    while (eingabe && (c = eingabe.get()) != '<')
        ;
    eingabe >> name._pos_in_list;

    while (eingabe && (c = eingabe.get()) != ',')
        ;
    eingabe >> name._id;

    while (eingabe && (c = eingabe.get()) != ',')
        ;
    eingabe >> name._list_type;

    while (eingabe && (c = eingabe.get()) != '>')
        ;

    return eingabe;
}
