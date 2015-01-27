/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

/*
 * The Class "InvTimer"
 * ------------------
 *
 * Description	:   this file contains the definition of the methods
 *		    introduced in the class "InvTimer"
 *
 * Date:	    29.03.1994
 *
 */

#include "InvTimer.h"
#include <time.h>

InvTimer::InvTimer(int sz)
{
    count_s = 0;
    count_e = 0;
    size = sz;
    before = new struct tms[size];
    after = new struct tms[size];
    startime = new clock_t[size];
    endtime = new clock_t[size];
}

InvTimer::~InvTimer()
{
    delete[] before;
    delete[] after;
    delete[] startime;
    delete[] endtime;
}

void InvTimer::print_to_file(char *filename, char *ModuleName, char *timerName)
{
    char *HostName = new char[256];

    time_t clk = time(NULL);

    if (gethostname(HostName, 100) == -1)
    {
        cerr << "can't get host name" << '\n';

        delete HostName;

        HostName = new char[strlen("HOST UNKNOWN") + 1];
        strcpy(HostName, "HOST UNKNOWN");
    }

    ofstream OutPrint(filename, ios::app); // open and write at the end of file
    if (!OutPrint)
    {
        error = 1;
        return;
    }

    OutPrint << "\n\n"
             << "====================================================" << '\n'
             << "Host name: " << HostName << '\n'
             << "Date: " << ctime(&clk) << '\n'
             << "Measurements of module: " << ModuleName << '\n'
             << "Description :" << timerName << '\n'
             << "Number of Measurements: " << size << "\n";

    OutPrint << "--------------------------------------------------------------------------" << '\n'
             << "No.  |    CPU time (sec)   |  Wall clock time (sec) |    Clock Ticks     |" << '\n'
             << "--------------------------------------------------------------------------" << '\n';
    /*       1                     2                        3		          4 */

    int lg = (count_s < count_e) ? count_s : count_e;
    for (int i = 0; i < lg; i++)
    {
        OutPrint << setw(5) << (i + 1)
                 << setw(21) << '|';
        //	OutPrint.setf(ios::fixed, ios::floatfield);
        OutPrint.precision(8);
        OutPrint << ((double)(after[i].tms_utime - before[i].tms_utime) / (double)CLK_TCK)
                 << setw(24) << '|'
                 << ((double)(endtime[i] - startime[i]) / (double)CLK_TCK)
                 << setw(20) << '|';
        OutPrint << (double)(endtime[i] - startime[i]) << '|'
                 << '\n';
    }
    OutPrint << " -------------------------------------------------------------------------" << '\n';
    OutPrint.close();
}
