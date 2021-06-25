/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "datacommand.hpp"

DataCommand::DataCommand(QUndoCommand *parent)
    : QUndoCommand(parent)
    , isUndone_(true)
    , isValid_(false)
{
}
