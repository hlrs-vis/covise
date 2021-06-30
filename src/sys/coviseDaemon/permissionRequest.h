/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_DAEMON_PERMISSION_REQUEST_H
#define COVISE_DAEMON_PERMISSION_REQUEST_H

#include <qtutil/NonBlockingDialogue.h>
#include <vrb/ProgramType.h>
class PermissionRequest : public covise::NonBlockingDialogue
{
    Q_OBJECT
public:
    PermissionRequest(vrb::Program program, int requestorId, const QString &description, QWidget* parent = nullptr);
    vrb::Program program() const;
    int requestorId() const;
signals:
    void permit(bool doPermit);

private:
    vrb::Program m_program;
    int m_requestorId;
};

#endif // COVISE_DAEMON_PERMISSION_REQUEST_H