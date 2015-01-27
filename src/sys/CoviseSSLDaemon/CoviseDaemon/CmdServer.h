/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

class CmdServer
{
public:
    CmdServer(void);
    ~CmdServer(void);

    void openServer();

public slots:
    void handlePortEvent(int);
};
