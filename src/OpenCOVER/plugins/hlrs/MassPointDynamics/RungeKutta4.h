/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RungeKutta4_h
#define RungeKutta4_h

#include <vector>

template <typename T, typename DT>
class RungeKutta4
{
public:
    RungeKutta4();

    RungeKutta4(T &);

    void integrate(double);

    void addState(T &);

protected:
    std::vector<T *> stateVector;
};

template <typename T, typename DT>
RungeKutta4<T, DT>::RungeKutta4()
{
}

template <typename T, typename DT>
RungeKutta4<T, DT>::RungeKutta4(T &s)
{
    stateVector.push_back(&s);
}

template <typename T, typename DT>
void RungeKutta4<T, DT>::addState(T &s)
{
    stateVector.push_back(&s);
}

template <typename T, typename DT>
void RungeKutta4<T, DT>::integrate(double h)
{
    double hh = h * 0.5;

    for (int i = 0; i < stateVector.size(); ++i)
    {
        DT dsN = stateVector[i]->dstate(0.0);

        T sA = *stateVector[i] + dsN * hh;
        DT dsA = sA.dstate(hh);

        T sB = *stateVector[i] + dsA * hh;
        DT dsB = sB.dstate(hh);

        T sC = *stateVector[i] + dsB * h;
        DT dsC = sC.dstate(h);

        *stateVector[i] += (dsN + dsA + dsA + dsB + dsB + dsC) * (h / 6.0);
    }
}

#endif
