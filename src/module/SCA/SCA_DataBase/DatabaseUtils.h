/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DATABASE_UTILS_H_
#define _DATABASE_UTILS_H_

#include <util/coviseCompat.h>
#include <util/coExport.h>
#include <vector>

class SCAEXPORT DatabaseUtils
{
public:
    static int readEmbossingResults(ifstream &emb_conn, ifstream &emb_displ,
                                    ifstream &emb_displ_exp,
                                    ifstream &emb_thick,
                                    vector<int> &epl, vector<int> &cpl,
                                    vector<float> &exc,
                                    vector<float> &eyc,
                                    vector<float> &ezc,
                                    vector<float> &dicke);

protected:
private:
    static int MarkCoords(vector<int> &cpl, std::vector<int> &mark);
    static int readDisplacements(ifstream &emb_displ, std::vector<int> &mark,
                                 vector<float> &exc, vector<float> &eyc, vector<float> &ezc,
                                 bool newFormat, int dickeSize);
    static int AddDisplacements(vector<float> &exc, vector<float> &eyc, vector<float> &ezc,
                                vector<float> &pxc, vector<float> &pyc, vector<float> &pzc,
                                std::vector<int> &mark);
};

#endif
