/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <util/coviseCompat.h>

#include "parser.h"
#include "DxObject.h"

void DxObject::setMember(char *&member, const char *value)
{
    delete member;
    if (NULL == value)
    {
        member = NULL;
    }
    else
    {
        member = new char[1 + strlen(value)];
        strcpy(member, value);
    }
}

void DxObject::setMember(char *&member, int number)
{
    delete member;
    member = new char[64];
    sprintf(member, "%d", number);
}

//For testing purposes only
void DxObject::show()
{
    cerr << "                   Name_: " << name_ << endl;
    cerr << "                   ObjectClass_:" << objectClass_ << endl;
    cerr << "                   DataOffset_:" << dataOffset_ << endl;
    cerr << "                   Rank_:" << rank_ << endl;
    cerr << "                   Items_:" << items_ << endl;
    cerr << "                   Type_:" << type_ << endl;
    cerr << "                   Shape_:" << shape_ << endl;
    cerr << "                   DataFormat_:" << dataFormat_ << endl;
    cerr << "                   ByteOrder_:" << byteOrder_ << endl;
    cerr << "                   FileName_:" << fileName_ << endl;
    cerr << "                   Follows_:" << follows_ << endl;
    cerr << "                   elementType_:" << elementType_ << endl;
    cerr << "                   Ref_: " << ref_ << endl;
    cerr << "                   Data_: " << data_ << endl;
    cerr << "                   Connections_: " << connections_ << endl;
    cerr << "                   Positions_: " << positions_ << endl;
}

DxObject::DxObject(int defaultByteOrder)
{
    name_ = NULL;
    objectClass_ = Parser::ARRAY;
    dataOffset_ = 0;
    rank_ = 0;
    items_ = 0;
    type_ = Parser::FLOAT;
    shape_ = 0;
    dataFormat_ = 0;
    byteOrder_ = defaultByteOrder;
    fileName_ = NULL;
    follows_ = false;
    elementType_ = NULL;
    ref_ = NULL;
    data_ = NULL;
    connections_ = NULL;
    positions_ = NULL;
    attributeDep_ = NULL;
    attributeName_ = NULL;
}
