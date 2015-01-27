/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ATTRIBUTE_CONTAINER_H
#define ATTRIBUTE_CONTAINER_H

#include <do/coDistributedObject.h>

#include <utility>
#include <string>
#include <vector>
#include <utility>
#include <list>
using namespace std;
using namespace covise;

class attributeContainer
{
public:
    attributeContainer(coDistributedObject *);
    virtual ~attributeContainer();
    string dummyName();
    int timeSteps();
    void clean();
    void addAttributes(coDistributedObject *,
                       vector<pair<string, string> > theseAttributes);
    /*
                                          vector< pair< string, string > >());
      */
private:
    string _dummyName;
    bool _isASet;
    int _timeSteps;
    vector<pair<string, string> > _primaryAttributes;
    list<pair<string, string> > _secondaryAttributes;
    vector<coDistributedObject *> _objects;
    void preOrderList(coDistributedObject *);
    void addSecondary(const coDistributedObject *);
};
#endif
