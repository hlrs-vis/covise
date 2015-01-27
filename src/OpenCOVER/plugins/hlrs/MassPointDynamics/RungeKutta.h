/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RungeKutta_h
#define RungeKutta_h

#include <vector>

template <typename S, typename L>
class RungeKutta
{
public:
    RungeKutta(std::vector<S> *, std::vector<L *> *);

    void integrate(double);

protected:
    std::vector<S> *stateVector;
    std::vector<L *> *linkVector;
};

template <typename S, typename L>
RungeKutta<S, L>::RungeKutta(std::vector<S> *sv, std::vector<L *> *lv)
    : stateVector(sv)
    , linkVector(lv)
{
}

template <typename S, typename L>
void RungeKutta<S, L>::integrate(double h)
{
    double hh = h * 0.5;
    unsigned int stateSize = stateVector->size();
    unsigned int linkSize = linkVector->size();

    for (unsigned int i = 0; i < linkSize; ++i)
    {
        (*linkVector)[i]->applyTo(stateVector);
    }
    std::vector<S> dsNVec(stateSize);
    std::vector<S> sAVec(stateSize);

    for (unsigned int j = 0; j < stateSize; ++j)
    {
        dsNVec[j] = (*stateVector)[j].dstate(h);
        sAVec[j] = (*stateVector)[j] + dsNVec[j] * hh;
    }

    for (unsigned int i = 0; i < linkSize; ++i)
    {
        (*linkVector)[i]->applyTo(&sAVec);
    }
    std::vector<S> dsAVec(stateSize);
    std::vector<S> sBVec(stateSize);
    for (unsigned int j = 0; j < stateSize; ++j)
    {
        dsAVec[j] = sAVec[j].dstate(h);
        sBVec[j] = (*stateVector)[j] + dsAVec[j] * hh;
    }

    for (unsigned int i = 0; i < linkSize; ++i)
    {
        (*linkVector)[i]->applyTo(&sBVec);
    }
    std::vector<S> dsBVec(stateSize);
    std::vector<S> sCVec(stateSize);
    for (unsigned int j = 0; j < stateSize; ++j)
    {
        dsBVec[j] = sBVec[j].dstate(h);
        sCVec[j] = (*stateVector)[j] + dsBVec[j] * hh;
    }

    for (unsigned int i = 0; i < linkSize; ++i)
    {
        (*linkVector)[i]->applyTo(&sCVec);
    }
    std::vector<S> dsCVec(stateSize);
    for (unsigned int j = 0; j < stateSize; ++j)
    {
        dsCVec[j] = sCVec[j].dstate(h);
    }

    for (unsigned int j = 0; j < stateSize; ++j)
    {
        (*stateVector)[j] += (dsNVec[j] + dsAVec[j] + dsAVec[j] + dsBVec[j] + dsBVec[j] + dsCVec[j]) * (h / 6.0);
    }

    /*for(int i=0; i<stateVector.size(); ++i) {
      S dsN = stateVector[i]->dstate(0.0);

      S sA = *stateVector[i] + dsN*hh;
      S dsA = sA.dstate(hh);

      S sB = *stateVector[i] + dsA*hh;
      S dsB = sB.dstate(hh);

      S sC = *stateVector[i] + dsB*h;
      S dsC = sC.dstate(h);

      *stateVector[i] += (dsN + dsA + dsA + dsB + dsB + dsC)*(h/6.0);
   }*/
}

#endif
