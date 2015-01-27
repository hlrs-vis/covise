/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coLabelMenuItem.h>

#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coColoredBackground.h>

using std::string;

namespace vrui
{

/** Constructor.
  @param name label string
*/
coLabelMenuItem::coLabelMenuItem(const string &name)
    : coRowMenuItem(name)
{
    container->addElement(label);
    background->setXAlignment(coUIContainer::CENTER);
}

/// Destructor
coLabelMenuItem::~coLabelMenuItem()
{
}

const char *coLabelMenuItem::getClassName() const
{
    return "colabelMenuItem";
}

bool coLabelMenuItem::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coRowMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
