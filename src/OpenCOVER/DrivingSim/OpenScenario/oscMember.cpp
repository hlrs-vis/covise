/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscMember.h>
#include <oscObjectBase.h>

using namespace OpenScenario;


/*oscMember::oscMember(std::string &n, oscMemberValue::MemberTypes t, oscObjectBase* owner, oscMemberValue *&mv): value(mv)
{
    name = n;
    type = t;
    owner->addMember(this);
}*/

oscMember::oscMember()
{
    value = NULL;
    owner = NULL;
    type = oscMemberValue::MemberTypes::INT;
}

oscMember::~oscMember()
{
}

void oscMember::registerWith(oscObjectBase* o)
{
    owner = o;
    owner->addMember(this);
}
