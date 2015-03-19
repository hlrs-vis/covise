/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Carbon.h"
#include "AtomStickInteractor.h"
#include "AtomBallInteractor.h"
Carbon::Carbon(string symbol, const char *interactorName, osg::Matrix m, float size, std::vector<osg::Vec3> connections, osg::Vec4 color)
    : Atom(symbol, interactorName, m, size, connections, color)
{
    if (connections.size() != 4)
        fprintf(stderr, "this is not a carbon\n");
}

// CH4
//
//   H
// H C H
//   H
bool
Carbon::isMethan()
{
    fprintf(stderr, "Carbon::isMethan...");
    int numH = 0;
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        AtomStickInteractor *conn = atomSticks_[i]->getConnectedStick();
        if (conn && (conn->getAtom()->getSymbol() == "H"))
        {
            numH++;
        }
    }
    if (numH == 4)
    {
        fprintf(stderr, "correct\n");
        return (true);
    }
    else
    {
        fprintf(stderr, "not correct\n");
        return (false);
    }
}
// C2H6 end-end
//
//   H H
// H C C H
//   H H
bool
Carbon::isEthan()
{
    if (isAlkaneEnd())
    {
        Carbon *nextCarbon = getNextCarbon(NULL);
        if (nextCarbon && nextCarbon->isAlkaneEnd())
            return (true);
        else
            return (false);
    }
    else
        return (false);
}

// C3H8 end-middle-end
//
//   H H H
// H C C C H
//   H H H
bool Carbon::isPropan()
{
    if (isAlkaneEnd())
    {
        Carbon *nextCarbon = getNextCarbon(NULL);
        Carbon *prevCarbon = this;
        if (nextCarbon && nextCarbon->isAlkaneMiddle())
        {
            nextCarbon = nextCarbon->getNextCarbon(prevCarbon);
            if (nextCarbon && nextCarbon->isAlkaneEnd())
                return (true);
            else
                return (false);
        }
        else
            return (false);
    }
    else
        return (false);
}
// C4H10 end-middle-middle-end
//   H H H H
// H C C C C H
//   H H H H
bool Carbon::isLinearButan()
{
    fprintf(stderr, "Carbon::isButan\n");
    if (isAlkaneEnd())
    {
        fprintf(stderr, "\t%s is end\n", atomBall_->getInteractorName());
        Carbon *prevCarbon = NULL; //NULL
        Carbon *currentCarbon = this; // NULL -C0
        Carbon *nextCarbon;
        for (int i = 0; i < 2; i++)
        {
            nextCarbon = currentCarbon->getNextCarbon(prevCarbon); // NULL-C0-C1
            if (nextCarbon && nextCarbon->isAlkaneMiddle())
            {
                fprintf(stderr, "\t%s is middle\n", nextCarbon->atomBall_->getInteractorName());
                prevCarbon = currentCarbon;
                currentCarbon = nextCarbon;

                nextCarbon = currentCarbon->getNextCarbon(prevCarbon); // NULL- C0- C1-C2
            }
            else
                return (false);
        }
        if (nextCarbon && nextCarbon->isAlkaneEnd()) // NULL-C0-C1-C2-NULL
        {
            fprintf(stderr, "\t%s is end\n", nextCarbon->atomBall_->getInteractorName());
            return (true);
        }
        else
            return (false);
    }
    else
        return (false);
}

bool Carbon::isLinearAlkane(int numC)
{
    //fprintf(stderr,"Carbon::isAlkane\n");
    if (isAlkaneEnd())
    {
        //fprintf(stderr,"\t%s is end\n", atomBall_->getInteractorName());
        Carbon *prevCarbon = NULL; //NULL
        Carbon *currentCarbon = this; // NULL -C0
        Carbon *nextCarbon;
        for (int i = 0; i < (numC - 2); i++)
        {
            nextCarbon = currentCarbon->getNextCarbon(prevCarbon); // NULL-C0-C1
            if (nextCarbon && nextCarbon->isAlkaneMiddle())
            {
                //fprintf(stderr,"\t%s is middle\n", nextCarbon->atomBall_->getInteractorName());
                prevCarbon = currentCarbon;
                currentCarbon = nextCarbon;

                nextCarbon = currentCarbon->getNextCarbon(prevCarbon); // NULL- C0- C1-C2
            }
            else
                return (false);
        }
        if (nextCarbon && nextCarbon->isAlkaneEnd()) // NULL-C0-C1-C2-NULL
        {
            //fprintf(stderr,"\t%s is end\n", nextCarbon->atomBall_->getInteractorName());
            return (true);
        }
        else
            return (false);
    }
    else
        return (false);
}

// endstueck eines alkans
//   H
// H C C
//   H
bool Carbon::isAlkaneEnd()
{
    int numH = 0;
    int numC = 0;

    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        AtomStickInteractor *conn = atomSticks_[i]->getConnectedStick();
        if (conn && conn->getAtom()->getSymbol() == "C")
        {
            numC++;
        }
        if (conn && conn->getAtom()->getSymbol() == "H")
        {
            numH++;
        }
    }

    if (numC == 1 && numH == 3)
    {
        fprintf(stderr, "Found Alkane End\n");
        return true;
    }
    else
    {
        return (false);
    }
}

Carbon *
Carbon::getNextCarbon(Carbon *prevCarbon)
{
    Carbon *nextCarbon = NULL;
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        AtomStickInteractor *conn = atomSticks_[i]->getConnectedStick();
        if (conn && conn->getAtom()->getSymbol() == "C")
        {
            nextCarbon = (Carbon *)(conn->getAtom());
            if (nextCarbon != prevCarbon)
                return (nextCarbon);
        }
    }
    return (NULL);
}

// mittelteil eines alkans
//   H
// C C C
//   H
bool Carbon::isAlkaneMiddle()
{
    int numH = 0;
    int numC = 0;

    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        AtomStickInteractor *conn = atomSticks_[i]->getConnectedStick();
        if (conn && conn->getAtom()->getSymbol() == "C")
        {
            numC++;
        }
        if (conn && conn->getAtom()->getSymbol() == "H")
        {
            numH++;
        }
    }
    if (numC == 2 && numH == 2)
    {
        fprintf(stderr, "Found Alkane Middle\n");
        return (true);
    }
    else
    {
        return (false);
    }
}

int Carbon::getNumAtoms(string symbol)
{
    int numH = 0;
    int numC = 1; // myself

    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        AtomStickInteractor *conn = atomSticks_[i]->getConnectedStick();
        if (conn && conn->getAtom()->getSymbol() == "C")
        {
            numC++;
        }
        if (conn && conn->getAtom()->getSymbol() == "H")
        {
            numH++;
        }
    }
    if (symbol == "C")
        return numC;
    else if (symbol == "H")
        return numH;
    else
    {
        fprintf(stderr, "this function does not check symbol %s\n", symbol.c_str());
        return (-1);
    }
}
