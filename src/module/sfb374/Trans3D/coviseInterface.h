/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_INTERFACE
#define COVISE_INTERFACE

class coviseInterface
{
public:
    coviseInterface();
    ~coviseInterface();
    void run(int, char **);
    void error(const char *);
    void warning(const char *);
    void info(const char *);

private:
};

extern coviseInterface covise;
#endif
