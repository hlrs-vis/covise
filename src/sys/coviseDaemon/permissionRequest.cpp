/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "permissionRequest.h"

PermissionRequest::PermissionRequest(vrb::Program program, int requestorId, const QString &description, QWidget* parent)
: covise::NonBlockingDialogue(parent)
, m_program(program)
, m_requestorId(requestorId)
{
	setWindowTitle("Application execution request");
	setQuestion("Do you want to execute this application?");
    setInfo(description);
	auto a = addOption("Allow");
	addOption("Decline");
    connect(this, &covise::NonBlockingDialogue::answer, this, [a, this](int option)
            { emit permit(option == a); });
}

vrb::Program PermissionRequest::program() const
{
    return m_program;
}

int PermissionRequest::requestorId() const
{
    return m_requestorId;
}


