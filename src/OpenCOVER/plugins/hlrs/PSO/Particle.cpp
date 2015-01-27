/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Author: Florian Seybold, Professor Gerhard Venter
 * University of Stellenbosch and High-Performance Computing Center Stuttgart
 * Copyright: 2007, 2008
*/

#include "Particle.h"

#include <string.h>
#include <stdio.h>
#ifdef _MSC_VER
#include <ymath.h>
#define INFINITY _FInf._Double
static int round(double d) { return ((int)(d + 0.5)); }
static int lround(double d) { return ((int)(d + 0.5)); }
#endif

using namespace pso;

int Particle::numpars = 0;
Particle **Particle::par = NULL;
Particle **Particle::bestpar = NULL;
double (*Particle::response)(double *) = NULL;
int Particle::nvar = 0;
double Particle::w = MAXINERTIA;
double Particle::c1 = 1.5;
double Particle::c2 = 2.5;
double *Particle::lowerbound = NULL;
double *Particle::upperbound = NULL;
bool *Particle::integer = NULL;
double Particle::gbestval = INFINITY;

double *Particle::gbestx = NULL;
double Particle::dt = 0;
double Particle::meanval = 0;
double Particle::devval = 0;
double Particle::meansortval = 0;
double Particle::devsortval = 0;
double *Particle::meanx = NULL;
double *Particle::devx = NULL;
/// Maximum number of particles.
const int Particle::MAXPARS = 100;

/// Maximum inertia.
const double Particle::MAXINERTIA = 1.4;
/// Minimum intertia.
const double Particle::MININERTIA = 0.35;
/// Fraction of particles (used by function computeSortValsMean() and computeSortValsDev() ).
const double Particle::FRACPARS = 0.20;
/// Fraction of inertia (used by function updateInertia() ).
const double Particle::FRACINERTIA = 0.975;
/// Covariance threshold of inertia.
const double Particle::COVTHRESHOLDINERTIA = 0.10;
/// Covariance threshold of craziness operator.
const double Particle::COVTHRESHOLDCRAZY = 1.0;

/// Defines the beta1 bound of the extend of the one dimensional search.
const double Particle::BOUNDBETA1 = 1.0;
/// Defines the beta2 bound of the extend of the one dimensional search.
const double Particle::BOUNDBETA2 = -1.0;

MTRand *Particle::mtrand = NULL;

Particle::Particle()
{
    particle = numpars;
    ++numpars;

    par[particle] = this;

    x = new double[nvar];
    v = new double[nvar];

    xbeta1 = new double[nvar];
    xbeta2 = new double[nvar];

    bestval = INFINITY;
    bestx = new double[nvar];

    for (int j = 0; j < nvar; ++j)
    {
        // Initial Position
        x[j] = rand() * (upperbound[j] - lowerbound[j]) + lowerbound[j];
        if (integer[j])
            x[j] = round(x[j]);
        // Initial Velocity
        v[j] = -(upperbound[j] - lowerbound[j]) + 2 * (upperbound[j] - lowerbound[j]) * rand();
    }
    //updateVal();
}

void Particle::init(double (*setresponse)(double *), int setnvar, double *setlowerbound, double *setupperbound, bool *setinteger, double setdt)
{
    initSwarm((*setresponse), setnvar, setlowerbound, setupperbound, setinteger, setdt);

    initRand();
}

void Particle::initSwarm(double (*setresponse)(double *), int setnvar, double *setlowerbound, double *setupperbound, bool *setinteger, double setdt)
{
    numpars = 0;
    par = NULL;
    bestpar = NULL;
    response = NULL;
    nvar = 0;
    w = MAXINERTIA;
    c1 = 1.5;
    c2 = 2.5;
    lowerbound = NULL;
    upperbound = NULL;
    gbestval = INFINITY;
    gbestx = NULL;
    dt = 0;
    meanval = 0;
    devval = 0;
    meansortval = 0;
    devsortval = 0;
    meanx = NULL;
    devx = NULL;

    par = new Particle *[MAXPARS];

    bestpar = new Particle *[MAXPARS];

    response = setresponse;

    nvar = setnvar;
    dt = setdt;

    gbestx = new double[setnvar];

    lowerbound = new double[setnvar];
    memcpy(lowerbound, setlowerbound, nvar * sizeof(double));

    upperbound = new double[setnvar];
    memcpy(upperbound, setupperbound, nvar * sizeof(double));

    integer = new bool[setnvar];
    memcpy(integer, setinteger, nvar * sizeof(bool));

    meanx = new double[setnvar];
    devx = new double[setnvar];
}

void Particle::destroy()
{
    delete[] lowerbound;
    delete[] upperbound;
    delete[] gbestx;
    delete[] integer;
    delete[] meanx;
    delete[] devx;

    destroyRand();
}

void Particle::all(void (Particle::*func)())
{
    for (int i = 0; i < numpars; ++i)
    {
        (par[i]->*func)();
    }
}

void Particle::updateVelocity()
{
    r1 = rand();
    r2 = rand();

    for (int j = 0; j < nvar; ++j)
    {
        v[j] = w * v[j] + c1 * r1 * (bestx[j] - x[j]) / dt + c2 * r2 * (gbestx[j] - x[j]) / dt;
    }
}

void Particle::updatePosition()
{
    for (int j = 0; j < nvar; ++j)
    {
        x[j] = x[j] + v[j] * dt;
    }

    enforceConstrains();
}

void Particle::enforceConstrains()
{
    for (int j = 0; j < nvar; ++j)
    {
        // Rounding if integer
        if (integer[j])
            x[j] = round(x[j]);

        // Fixing bounds
        if (x[j] < lowerbound[j])
        {
            x[j] = lowerbound[j];
            v[j] = 0.0;
        }
        else if (x[j] > upperbound[j])
        {
            x[j] = upperbound[j];
            v[j] = 0.0;
        }
    }
}

void Particle::updateVal()
{

    val = (*response)(x);

    Particle::updateBestVal();
}

void Particle::updateBestVal()
{
    if (val < bestval)
    {
        bestval = val;
        memcpy(bestx, x, nvar * sizeof(double));
    }

    if (bestval < gbestval)
    {
        gbestval = bestval;
        memcpy(gbestx, bestx, nvar * sizeof(double));
    }
}

int Particle::getNumPar()
{
    return particle;
}

double Particle::getVal()
{
    return val;
}

void Particle::setVal(double setVal)
{
    val = setVal;
}

double Particle::getBestVal()
{
    return bestval;
}

double *Particle::getX()
{
    return x;
}

double *Particle::getV()
{
    return v;
}

double *Particle::getBestX()
{
    return bestx;
}

void Particle::updateInertia()
{
    sortVals();

    computeSortValsMean();
    computeSortValsDev();

    if (fabs(meansortval) < 1.0)
    {
        meansortval = (meansortval >= 0.0) ? 1.0 : -1.0;
    }

    double cov = fabs(devsortval / meansortval);
    double locw;

    if (cov <= COVTHRESHOLDINERTIA)
    {
        locw = FRACINERTIA * w;
        if (locw < MININERTIA)
            locw = MININERTIA;
        w = locw;
        std::cerr << "Set new inertia: " << w << std::endl;
    }
}

void Particle::goCrazy()
{
    sortVals();

    computeValsMean();
    computeValsDev();

    if (fabs(meanval) < 1.0)
    {
        meanval = (meanval >= 0.0) ? 1.0 : -1.0;
    }

    double cov = fabs(devval / meanval);

    if (cov <= COVTHRESHOLDCRAZY)
    {
        computeXMean();
        computeXDev();

        int startbestpar = 0;
        while ((bestpar[startbestpar]->val <= meanval + 0.5 * devval))
        {
            ++startbestpar;
            if (startbestpar >= numpars)
                break;
        }

        double minx, dx;

        for (int j = 0; j < nvar; ++j)
        {
            minx = meanx[j] - devx[j];
            dx = 2 * devx[j];

            for (int i = startbestpar; i < numpars; ++i)
            {
                bestpar[i]->x[j] = minx + rand() * dx;
                bestpar[i]->v[j] = rand() * (bestpar[i]->bestx[j] - bestpar[i]->x[j]);
                //if(bestpar[i]->x[j] == 0) {std::cerr << "Zero!"; exit(1); }
                //if(bestpar[i]->v[j] == 0) {std::cerr << "Zero!"; exit(1); }
                bestpar[i]->enforceConstrains();
            }
        }

        std::cerr << "Particles gone crazy: " << (numpars - startbestpar) << std::endl;
    }
}

double Particle::getGBestVal()
{
    return gbestval;
}

double *Particle::getGBestX()
{
    return gbestx;
}

Particle **Particle::getPars()
{
    return par;
}

Particle **Particle::getBestPars()
{
    return bestpar;
}

void Particle::computeValsMean()
{
    meanval = 0;

    for (int i = 0; i < numpars; ++i)
        meanval += par[i]->val;

    meanval = meanval / numpars;
}

void Particle::computeValsDev()
{
    devval = 0;

    for (int i = 0; i < numpars; ++i)
        devval += pow((par[i]->val - meanval), 2);

    devval = sqrt(devval / numpars);
}

void Particle::computeSortValsMean()
{
    int usepars = lround(FRACPARS * numpars);

    meansortval = 0;

    for (int i = 0; i < usepars; ++i)
        meansortval += bestpar[i]->val;

    meansortval = meansortval / usepars;
}

void Particle::computeSortValsDev()
{
    int usepars = lround(FRACPARS * numpars);

    devsortval = 0;

    for (int i = 0; i < usepars; ++i)
        devsortval += pow((bestpar[i]->val - meansortval), 2);

    devsortval = sqrt(devsortval / usepars);
}

void Particle::computeXMean()
{
    for (int j = 0; j < nvar; ++j)
    {
        meanx[j] = 0;

        for (int i = 0; i < numpars; ++i)
            meanx[j] += par[i]->x[j];

        meanx[j] = meanx[j] / numpars;
    }
}

void Particle::computeXDev()
{
    for (int j = 0; j < nvar; ++j)
    {
        devx[j] = 0;

        for (int i = 0; i < numpars; ++i)
            devx[j] += pow((par[i]->x[j] - meanx[j]), 2);

        devx[j] = sqrt(devx[j] / numpars);
    }
}

void Particle::sortVals()
{ //insertion sort
    Particle *temppar;
    int j;

    memcpy(bestpar, par, numpars * sizeof(Particle **));

    for (int i = 0; i < numpars; ++i)
    {
        temppar = bestpar[i];
        for (j = i - 1; (j >= 0) && (bestpar[j]->val > temppar->val); --j)
            bestpar[j + 1] = bestpar[j];
        bestpar[j + 1] = temppar;
    }
}

void Particle::updateBeta()
{
    bool ok = false;
    double reduct = 0.8;

    beta1 = BOUNDBETA1;
    beta2 = BOUNDBETA2;

    for (int j = 0; j < nvar; ++j)
    {
        if (integer[j])
        {
            xbeta1[j] = x[j];
            xbeta2[j] = x[j];
        }
        else
        {
            ok = false;
            while (!ok)
            {
                xbeta1[j] = x[j] + v[j] * beta1;
                xbeta2[j] = x[j] + v[j] * beta2;

                //std::cerr << "xbeta1[" << j << "] = " << xbeta1[j] << "\t";
                //std::cerr << "xbeta2[" << j << "] = " << xbeta2[j] << std::endl;

                if (((xbeta1[j] < lowerbound[j]) || (xbeta2[j] > upperbound[j]))
                    || ((xbeta2[j] < lowerbound[j]) || (xbeta1[j] > upperbound[j])))
                {
                    beta1 *= reduct;
                    beta2 *= reduct;
                }
                else
                {
                    ok = true;
                }
            }
        }
    }
    /*
	for(int j=0; j<nvar; ++j) {
		if(!integer[j]) {
			if((xbeta1[j] < lowerbound[j]) || (xbeta1[j] > upperbound[j])) {
				beta1 = 0.5 * beta2;

				xbeta1[j] = x[j] + v[j]*beta1;
				xbeta2[j] = x[j] + v[j]*beta2;

				break;
			}
			else if((xbeta2[j] < lowerbound[j]) || (xbeta2[j] > upperbound[j])) {
				beta2 = 0.5 * beta1;

				xbeta1[j] = x[j] + v[j]*beta1;
				xbeta2[j] = x[j] + v[j]*beta2;

				break;
			}
		}
	}	
	*/
    //std::cerr << "Particle " << particle << ":\t beta1 = " << beta1 << "\t beta2 = " << beta2 << std::endl;
}

void Particle::updateValBeta()
{
    valbeta1 = (*response)(xbeta1);
    valbeta2 = (*response)(xbeta2);
}

void Particle::setValBeta1(double setVal)
{
    valbeta1 = setVal;
}

void Particle::setValBeta2(double setVal)
{
    valbeta2 = setVal;
}

void Particle::approximateX()
{
    double beta;

    double a = (beta2 / beta1 * (valbeta1 - val) - valbeta2 + val) / (beta2 * beta1 - beta2 * beta2);
    double b = (beta2 * beta2 / (beta1 * beta1) * (valbeta1 - val) - valbeta2 + val) / (beta2 * beta2 / beta1 - beta2);
    //std::cerr << "a: " << a << "\t" << "b: " << b << "\t" << beta1 << "\t" << beta2 << "\t" << valbeta1 << "\t" << valbeta2 << "\t";
    if (a <= 0)
    {
        //double valBB1 = a*BOUNDBETA1*BOUNDBETA1 + b*BOUNDBETA1 + val;
        //double valBB2 = a*BOUNDBETA2*BOUNDBETA2 + b*BOUNDBETA2 + val;
        double valBB1 = a * beta1 * beta1 + b * beta1 + val;
        double valBB2 = a * beta2 * beta2 + b * beta2 + val;

        if (valBB1 < valBB2)
        {
            //beta = BOUNDBETA1;
            beta = beta1;
        }
        else
        {
            //beta = BOUNDBETA2;
            beta = beta2;
        }
    }
    else
    {
        beta = -b / (2 * a);
    }
    //std::cerr << "Beta: " << beta << std::endl;
    for (int j = 0; j < nvar; ++j)
    {
        //if(!integer[j])
        x[j] = x[j] + v[j] * dt * beta;
    }

    enforceConstrains();
}

void Particle::approximateV()
{
    double beta;

    double a = (beta2 / beta1 * (valbeta1 - val) - valbeta2 + val) / (beta2 * beta1 - beta2 * beta2);
    double b = (beta2 * beta2 / (beta1 * beta1) * (valbeta1 - val) - valbeta2 + val) / (beta2 * beta2 / beta1 - beta2);
    //std::cerr << "a: " << a << "\t" << "b: " << b << "\t" << beta1 << "\t" << beta2 << "\t" << valbeta1 << "\t" << valbeta2 << "\t";
    if (a <= 0)
    {
        //double valBB1 = a*BOUNDBETA1*BOUNDBETA1 + b*BOUNDBETA1 + val;
        //double valBB2 = a*BOUNDBETA2*BOUNDBETA2 + b*BOUNDBETA2 + val;
        double valBB1 = a * beta1 * beta1 + b * beta1 + val;
        double valBB2 = a * beta2 * beta2 + b * beta2 + val;

        if (valBB1 < valBB2)
        {
            //beta = BOUNDBETA1;
            beta = beta1;
        }
        else
        {
            //beta = BOUNDBETA2;
            beta = beta2;
        }
    }
    else
    {
        beta = -b / (2 * a);
    }
    //std::cerr << "Beta: " << beta << std::endl;
    for (int j = 0; j < nvar; ++j)
    {
        //if(!integer[j])
        v[j] = v[j] * beta;
    }
}

double *Particle::getPositionBeta1()
{
    return xbeta1;
}

double *Particle::getPositionBeta2()
{
    return xbeta2;
}

void Particle::printState(std::ostream &os) const
{
    os << val;

    for (int i = 0; i < nvar; ++i)
        os << "\t" << x[i];
    for (int i = 0; i < nvar; ++i)
        os << "\t" << v[i];
}

void Particle::openParticleStateFile(std::ofstream &psofs)
{
    std::stringstream psofns;
    std::string psofn;

    psofns << "particle." << std::setfill('0') << std::setw(3) << particle;
    psofns >> psofn;
    psofs.open(psofn.c_str());
}

void Particle::initRand()
{
    //struct timeval acttime;
    //gettimeofday( &acttime, NULL );

    //mtrand = new MTRand(acttime.tv_sec + acttime.tv_usec);
    mtrand = new MTRand();
}
void Particle::destroyRand()
{
    delete mtrand;
}
double Particle::rand()
{
    return mtrand->rand();
}

int pso::griewank5_nvar = 5;
double pso::griewank5_lbound[5] = { -600, -600, -600, -600, -600 };
double pso::griewank5_ubound[5] = { 600, 600, 600, 600, 600 };
bool pso::griewank5_integer[5] = { false, false, false, false, false };
double pso::griewank5(double *x)
{
    double sum = 0.0;
    double product = 1.0;

    for (int i = 0; i < griewank5_nvar; ++i)
    {
        sum += x[i] * x[i] / 4000;
        product *= cos(x[i] / sqrt((double)i + 1));
    }

    return (sum - product + 1);
}

int pso::griewank3_nvar = 3;
double pso::griewank3_lbound[3] = { -600, -600, -600 };
double pso::griewank3_ubound[3] = { 600, 600, 600 };
bool pso::griewank3_integer[3] = { false, false, false };
double pso::griewank3(double *x)
{
    double sum = 0.0;
    double product = 1.0;

    for (int i = 0; i < griewank3_nvar; ++i)
    {
        sum += x[i] * x[i] / 4000;
        product *= cos(x[i] / sqrt((double)i + 1));
    }

    return (sum - product + 1);
}

int pso::griewank2_nvar = 2;
double pso::griewank2_lbound[2] = { -600, -600 };
double pso::griewank2_ubound[2] = { 600, 600 };
bool pso::griewank2_integer[2] = { false, false };
double pso::griewank2(double *x)
{
    double sum = 0.0;
    double product = 1.0;

    for (int i = 0; i < griewank2_nvar; ++i)
    {
        sum += x[i] * x[i] / 4000;
        product *= cos(x[i] / sqrt((double)i + 1));
    }

    return (sum - product + 1);
}

int pso::rosenbrock3_nvar = 3;
double pso::rosenbrock3_lbound[3] = { -2.048, -2.048, -2.048 };
double pso::rosenbrock3_ubound[3] = { 2.048, 2.048, 2.048 };
bool pso::rosenbrock3_integer[3] = { false, false, false };
double pso::rosenbrock3(double *x)
{
    double sum = 0.0;
    for (int i = 0; i < (rosenbrock3_nvar - 1); ++i)
    {
        sum += (pow(1 - x[i], 2) + 100 * pow(x[i + 1] - pow(x[i], 2), 2));
    }

    return sum;
}

int pso::rosenbrock2_nvar = 2;
double pso::rosenbrock2_lbound[2] = { -2.048, -2.048 };
double pso::rosenbrock2_ubound[2] = { 2.048, 2.048 };
bool pso::rosenbrock2_integer[2] = { false, false };
double pso::rosenbrock2(double *x)
{
    return 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1 - x[0]) * (1 - x[0]);
}
