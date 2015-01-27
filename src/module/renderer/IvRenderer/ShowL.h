/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: mimics an assoc array <char *,int>                     ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 06.04.2001                                                ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef ShowL_H
#define ShowL_H

class ShowL
{
public:
    ShowL();
    // stores the entry corresponding to nm in ret
    int get(const char *nm, int &ret);
    // sets the entry corresponding to nm to val
    void add(const char *nm, const int &val);
    // removes the entry nm
    int remove(const char *nm);
    // same as add but only for COVISE obj names which will be stored as
    // reduced names
    void addCoObjNm(const char *nm, const int &val);
    // stores the entry corresponding to nm in ret. If it assumes nm to be a
    // COVISE obj name and applies reduce(..) first. If this returns nothingn
    // it tries nm as row name.
    int getCoObjNm(const char *nm, int &ret);

    void removeAll();

    ~ShowL();

private:
    int numEn_;
    int incAlloc_;
    int numAlloc_;

    char **arr_;
    int *val_;
    // reduce COVISE obj names (XXX_Y_OUT_ABC -> XXX_Y_OUT)
    // returns the length of the str
    int reduce(const char *nm, char *redNm);
};
#endif
