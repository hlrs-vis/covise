/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_RUN_REC_LIST_H_
#define _CO_RUN_REC_LIST_H_

// 28.03.99

#include "coDLList.h"
#include "coObjID.h"
#include <iostream>

#ifndef INLINE
#define INLINE inline
#endif

namespace covise
{

// structure identifying a 'run' in the module network
// -> change coSendBuffer/coRecvBuffer when changing structure
struct coRunRec
{
    int timeStep, blockNo, numBlocks; // include numBlocks to identify different block structure
    int runID; // ID, sequentially counted up per execution
    // on MPP internal: node number
    coRunRec(const coObjInfo &info)
    {
        timeStep = info.timeStep;
        blockNo = info.blockNo;
        numBlocks = info.numBlocks;
        runID = -1;
    }
    coRunRec(){};
    bool operator!=(const coRunRec &r)
    {
        return (r.blockNo != blockNo || r.timeStep != timeStep || r.numBlocks != numBlocks);
    }
};

INLINE std::ostream &operator<<(std::ostream &str, const coRunRec &rec)
{
    str << "RunRec: T=" << rec.timeStep
        << " Block #" << rec.blockNo
        << "/" << rec.numBlocks
        << " RunID=" << rec.runID;
    return str;
}

// compare function
INLINE bool operator==(const coRunRec &step1, const coRunRec &step2)
{
    return (step1.timeStep == step2.timeStep)
           && (step1.blockNo == step2.blockNo)
           && (step1.numBlocks == step2.numBlocks);
}

/**
 * Class to handle lists of runRec's
 * 
 */
#ifndef _WIN32
class UTILEXPORT coRunRecList : public coDLList<coRunRec>
#else
class coRunRecList : public coDLList<coRunRec>
#endif
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coRunRecList(const coRunRecList &);

    /// Assignment operator: NOT  IMPLEMENTED
    coRunRecList &operator=(const coRunRecList &);

public:
    typedef coDLListIter<coRunRec> Iter;

    /// Default constructor: NOT  IMPLEMENTED
    coRunRecList(){};

    /// Destructor
    virtual ~coRunRecList(){};

    /// add a record, replace existing equal ones
    void add(const coRunRec &rec);

    /// get element with certain ID
    Iter getID(int runID);

    /// get element with certain Contents
    Iter getElem(const coRunRec &rec);
};

INLINE std::ostream &operator<<(std::ostream &str, coRunRecList &list)
{
    coRunRecList::Iter rec(list);
    while (rec)
    {
        str << *rec << std::endl;
        rec++;
    }
    return str;
}

// structure needed on MPP to combine RunRec's and Nodes
struct coMppRun
{
    coRunRec runRec;
    int node;

    coMppRun(const coObjInfo &info)
    {
        runRec.timeStep = info.timeStep;
        runRec.blockNo = info.blockNo;
        runRec.numBlocks = info.numBlocks;
        runRec.runID = -1;
        node = 0;
    }

    coMppRun(const coRunRec &old)
    {
        runRec = old;
        node = 0;
    }

    coMppRun(){};

    bool notFit(const coObjInfo &info) const
    {
        return (runRec.timeStep != info.timeStep)
               || (runRec.blockNo != info.blockNo)
               || (runRec.numBlocks != info.numBlocks);
    }

    bool notFit(const coRunRec &rec) const
    {
        return (runRec.timeStep != rec.timeStep)
               || (runRec.blockNo != rec.blockNo)
               || (runRec.numBlocks != rec.numBlocks);
    }

    bool notFit(const coMppRun &old) const
    {
        return (runRec.timeStep != old.runRec.timeStep)
               || (runRec.blockNo != old.runRec.blockNo)
               || (runRec.numBlocks != old.runRec.numBlocks);
    }
};

/// get element with certain ID
INLINE coRunRecList::Iter coRunRecList::getID(int runID)
{
    Iter rec(*this);
    while ((rec) && ((*rec).runID != runID))
        rec++;
    return rec;
}

/// get element with certain Contents
INLINE coRunRecList::Iter coRunRecList::getElem(const coRunRec &searchRec)
{
    Iter rec(*this);
    while ((rec) && !((*rec) == searchRec))
        rec++;
    return rec;
}

/// add a record, replace existing equal ones
INLINE void coRunRecList::add(const coRunRec &addRec)
{

    Iter rec(*this);
    while ((rec) && !((*rec) == addRec))
    {
        // if we got it, we needn't appeand it twice
        if ((*rec) == addRec)
            return;
        rec++;
    }
    append(addRec);
}
}

#endif
