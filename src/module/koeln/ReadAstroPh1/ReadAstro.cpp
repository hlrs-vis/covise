/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
    \brief  read star data of 1st physical institute
 
    \author Martin Aumueller <aumueller@uni-koeln.de>
    \author (C) 2007 ZAIK, University of Cologne, Germany
 
    \date   12.12.2007
 */

#include <do/coDoSet.h>
#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include <api/coModule.h>
#include <climits>
#include <cfloat>
#include <cassert>
#include <map>
#include <string>
#include <netcdfcpp.h>

using namespace covise;
float minname = FLT_MAX, maxname = -FLT_MAX;

using namespace covise;

class ReadAstro : public coModule
{
public:
    ReadAstro(int argc, char *argv[]);

private:
    // ports:
    coOutputPort *m_outPoint;
    coOutputPort *m_outType;
    coOutputPort *m_outMass;
    coOutputPort *m_outLogMass;
    coOutputPort *m_outVelocity;
    coOutputPort *m_outForce;
    coOutputPort *m_outForceDot;
    coOutputPort *m_outNames;
    coOutputPort *m_outDummies;
    coOutputPort *m_outDiskMass;
    coOutputPort *m_outDiskMomentum;

    // parameters:
    coFileBrowserParam *m_paramFilename; ///< name of NetCDF file
    coFloatParam *m_paramStep; ///< step width for interpolating time steps

    // methods
    virtual int compute(const char *port);
    bool open(const char *netcdfpath);
    bool read(int name, float time);
    bool close();

    // data
    NcFile *m_ncfile;
    NcVar *m_ncdata;
    int m_currentIndex;

    struct Star
    {
        float t;
        float step;
        float x[3];
        float v[3];
        float f[3];
        float fdot[3];
        float name;
        float i;
        float type;
        float dummy0;
        float mass;
        float logmass;
        float momentum;
        float diskmass;
    };
    struct StarTime : public std::deque<Star>
    {
        StarTime()
            : done(true)
        {
        }
        bool done;
    };
    std::vector<StarTime> m_stars;
};

/// Constructor
ReadAstro::ReadAstro(int argc, char *argv[])
    : coModule(argc, argv, "Read Astro files containing lists of atom positions and their element types for several time steps.")
    , m_ncfile(NULL)
    , m_ncdata(NULL)
    , m_currentIndex(0)
{

    // Create ports:
    m_outPoint = addOutputPort("Location", "Points", "Star location");
    m_outType = addOutputPort("Type", "Float", "Star type");
    m_outMass = addOutputPort("Mass", "Float", "Star mass");
    m_outLogMass = addOutputPort("LogMass", "Float", "Logarithmic star mass");
    m_outVelocity = addOutputPort("Velocity", "Vec3", "Star velocity");
    m_outForce = addOutputPort("Force", "Vec3", "Force");
    m_outForceDot = addOutputPort("ForceDot", "Vec3", "Derivative of force");
    m_outNames = addOutputPort("Name", "Float", "Name/index");
    m_outDummies = addOutputPort("Dummy0", "Float", "Dummy0");
    m_outDiskMomentum = addOutputPort("DiskMomentum", "Float", "Disk momentum");
    m_outDiskMass = addOutputPort("DiskMass", "Float", "Disk mass");

    // Create parameters:
    m_paramFilename = addFileBrowserParam("Filename", "NetCDF file");
    m_paramFilename->setValue("data/", "*.nc;*.NC/*");

    m_paramStep = addFloatParam("Timestep", "animation time step");
    m_paramStep->setValue(0.1);
}

bool ReadAstro::close()
{
    //delete m_ncdata;
    m_ncdata = NULL;

    delete m_ncfile;
    m_ncfile = NULL;
    return true;
}

bool ReadAstro::open(const char *path)
{
    assert(!m_ncfile);
    m_ncfile = new NcFile(path, NcFile::ReadOnly);
    if (!m_ncfile->is_valid())
    {
        close();
        sendError("failed to open NetCDF file %s", path);
        return false;
    }

    if (m_ncfile->get_format() == NcFile::BadFormat)
    {
        close();
        sendError("bad NetCDF file");
        return false;
    }

    fprintf(stderr, "dims=%d, vars=%d, attrs=%d\n",
            m_ncfile->num_dims(), m_ncfile->num_vars(), m_ncfile->num_atts());

    for (int i = 0; i < m_ncfile->num_dims(); ++i)
    {
        fprintf(stderr, "%s: %ld\n",
                m_ncfile->get_dim(i)->name(),
                m_ncfile->get_dim(i)->size());
    }
    for (int i = 0; i < m_ncfile->num_vars(); ++i)
    {
        fprintf(stderr, "%s: dims=%d atts=%d vals=%ld type=%d\n",
                m_ncfile->get_var(i)->name(),
                m_ncfile->get_var(i)->num_dims(),
                m_ncfile->get_var(i)->num_atts(),
                m_ncfile->get_var(i)->num_vals(),
                m_ncfile->get_var(i)->type());
        //int dims = m_ncfile->get_var(i)->num_dims();
        NcVar *var = m_ncfile->get_var(i);
        for (int j = 0; j < var->num_dims(); ++j)
        {
            fprintf(stderr, "   %s: %ld edge=%ld\n",
                    var->get_dim(j)->name(),
                    var->get_dim(j)->size(),
                    var->edges()[j]);
        }
    }
    for (int i = 0; i < m_ncfile->num_atts(); ++i)
    {
        fprintf(stderr, "%s\n", m_ncfile->get_att(i)->name());
    }

    m_ncdata = m_ncfile->get_var("data");
    if (!m_ncdata)
    {
        close();
        sendError("\"data\" variable not found");
        return false;
    }
    if (m_ncdata->num_dims() != 3)
    {
        close();
        sendError("\"data\" variable does not have 3 dimensions but %d", m_ncdata->num_dims());
        return false;
    }
    NcDim *coldim = m_ncdata->get_dim(2);
    if (!coldim || strcmp(coldim->name(), "cols"))
    {
        close();
        sendError("\"data\" variable does not have \"cols\" dimension");
        return false;
    }
    if (coldim->size() != 22)
    {
        close();
        sendError("\"cols\" dimension has value %ld instead of 22", coldim->size());
        return false;
    }
    if (m_ncdata->type() != ncFloat)
    {
        close();
        sendError("\"data\" does not have ncFloat type");
        return false;
    }

    return true;
}

bool ReadAstro::read(int name, float time)
{
    fprintf(stderr, "reading for no %d, time=%f\n", name, time);
    bool done = false;
    for (int i = m_currentIndex; i < m_ncdata->get_dim(0)->size(); ++i)
    {
        for (int j = 0; j < m_ncdata->get_dim(1)->size(); ++j)
        {
            union
            {
                float val[22];
                struct Star s;
            } d;
            m_ncdata->get(d.val, 1, 1, 22);
            //fprintf(stderr, "name=%f", d.s.name);
            int curname = int(d.s.name + 0.5f);
            if (d.s.name < minname)
                minname = d.s.name;
            if (d.s.name > maxname)
                maxname = d.s.name;
#ifdef DEBUG
            if (curname == 0)
                fprintf(stderr, "name ist 0: i=%f\n", d.s.i);
            if (curname == 1 || abs(float(curname) - d.s.name) > 0.001)
            {
                fprintf(stderr, "t\tstep\tx\ty\tz\tvx\tvy\tvz\tname\tindex\tmass\n");
                fprintf(stderr, "%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\n",
                        d.s.t, d.s.step, d.s.x[0], d.s.x[1], d.s.x[2], d.s.v[0], d.s.v[1], d.s.v[2], d.s.name, d.s.i, d.s.mass);
            }
#endif
            int num = m_stars.size();

            if (curname >= num)
            {
                m_stars.resize(curname + 1);
#ifdef DEBUG
                fprintf(stderr, "%d -> %d\n", num, (int)m_stars.size());
#endif
            }
            if (curname != 1 || d.s.name == 1.f)
            {
                m_stars[curname].push_back(d.s);
                m_stars[curname].done = false;
            }

            //fprintf(stderr, " %f", d.s.t);

            m_ncdata->set_cur(i, j, 0);
            //fprintf(stderr,"\n");

            if (curname == name && d.s.t >= time)
            {
                done = true;
            }
        }
        if (done)
        {
            m_currentIndex = i + 1;
#ifdef DEBUG
            fprintf(stderr, "read, currentIndex now at %d\n", m_currentIndex);
#endif
            return true;
        }
    }
    return false;
}

/// Compute routine: load checkpoint file
int ReadAstro::compute(const char *)
{
    const char *path = m_paramFilename->getValue();
    if (!open(path))
    {
        return STOP_PIPELINE;
    }

    std::vector<coDoPoints *> points;
    std::vector<coDoFloat *> types;
    std::vector<coDoFloat *> masses;
    std::vector<coDoFloat *> logmasses;
    std::vector<coDoVec3 *> velocities;
    std::vector<coDoVec3 *> forces;
    std::vector<coDoVec3 *> forcedots;
    std::vector<coDoFloat *> simNames;
    std::vector<coDoFloat *> dummies;
    std::vector<coDoFloat *> diskmasses;
    std::vector<coDoFloat *> diskmomentums;
    float time = 0.f;
    int timestep = 0;
    bool eof = !read(1, 1.f);
    bool empty = false;
    fprintf(stderr, "eof=%d\n", int(eof));
    while (!eof || !empty)
    {
        empty = true;
        std::stringstream name;

        for (int i = 0; i < m_stars.size(); ++i)
        {
            // read additional data if necessary
            if (m_stars[i].empty() || m_stars[i].back().t < time)
            {
                if (!eof)
                    eof = !read(i, time);
                if (!m_stars[i].empty() && m_stars[i].back().t >= time)
                    empty = false;
            }
            else
                empty = false;

            // throw out data which is not necessary any longer
            for (StarTime::iterator it = m_stars[i].begin();
                 it != m_stars[i].end();
                 ++it)
            {
                StarTime::iterator next = it;
                ++next;
                if (next == m_stars[i].end())
                    break;

                if ((*next).t < time)
                {
                    m_stars[i].pop_front();
#ifdef DEBUG
                    if (m_stars[i].size() < 5)
                        fprintf(stderr, "discarding: t=%f, %i (%f), left: %d\n", time, i, (*next).t, int(m_stars[i].size()));
#endif
                }
                else
                    break;
            }
        }

        int nstars = 0;
        for (int i = 0; i < m_stars.size(); ++i)
        {
            if (m_stars[i].size() >= 1)
                ++nstars;
        }

        name.clear();
        name << m_outPoint->getObjName() << "_" << timestep;
        coDoPoints *p = new coDoPoints(name.str().c_str(), nstars);
        points.push_back(p);
        float *x[3];
        p->getAddresses(&x[0], &x[1], &x[2]);

        name.clear();
        name << m_outType->getObjName() << "_" << timestep;
        coDoFloat *type = new coDoFloat(name.str().c_str(), nstars);
        types.push_back(type);
        float *t;
        type->getAddress(&t);

        name.clear();
        name << m_outMass->getObjName() << "_" << timestep;
        coDoFloat *mass = new coDoFloat(name.str().c_str(), nstars);
        masses.push_back(mass);
        float *m;
        mass->getAddress(&m);

        name.clear();
        name << m_outLogMass->getObjName() << "_" << timestep;
        coDoFloat *logmass = new coDoFloat(name.str().c_str(), nstars);
        logmasses.push_back(logmass);
        float *lm;
        logmass->getAddress(&lm);

        name.clear();
        name << m_outDiskMass->getObjName() << "_" << timestep;
        coDoFloat *diskmass = new coDoFloat(name.str().c_str(), nstars);
        diskmasses.push_back(diskmass);
        float *dm;
        diskmass->getAddress(&dm);

        name.clear();
        name << m_outDiskMomentum->getObjName() << "_" << timestep;
        coDoFloat *diskmomentum = new coDoFloat(name.str().c_str(), nstars);
        diskmomentums.push_back(diskmomentum);
        float *mom;
        diskmomentum->getAddress(&mom);

        name.clear();
        name << m_outVelocity->getObjName() << "_" << timestep;
        coDoVec3 *velocity = new coDoVec3(name.str().c_str(), nstars);
        velocities.push_back(velocity);
        float *v[3];
        velocity->getAddresses(&v[0], &v[1], &v[2]);

        name.clear();
        name << m_outForce->getObjName() << "_" << timestep;
        coDoVec3 *force = new coDoVec3(name.str().c_str(), nstars);
        forces.push_back(force);
        float *f[3];
        force->getAddresses(&f[0], &f[1], &f[2]);

        name.clear();
        name << m_outForceDot->getObjName() << "_" << timestep;
        coDoVec3 *forcedot = new coDoVec3(name.str().c_str(), nstars);
        forcedots.push_back(forcedot);
        float *fdot[3];
        forcedot->getAddresses(&fdot[0], &fdot[1], &fdot[2]);

        name.clear();
        name << m_outNames->getObjName() << "_" << timestep;
        coDoFloat *simName = new coDoFloat(name.str().c_str(), nstars);
        simNames.push_back(simName);
        float *fname = simName->getAddress();

        name.clear();
        name << m_outDummies->getObjName() << "_" << timestep;
        coDoFloat *dummy = new coDoFloat(name.str().c_str(), nstars);
        dummies.push_back(dummy);
        float *fdummy = dummy->getAddress();

        int n = 0;
        for (int i = 0; i < m_stars.size(); ++i)
        {
            // store data
            switch (m_stars[i].size())
            {
            case 0:
                break;
            case 1:
            {
                const Star &s = m_stars[i].front();
                const float dt = time - s.t;
                for (int j = 0; j < 3; ++j)
                {
                    x[j][n] = s.x[j] + s.v[j] * dt;
                    v[j][n] = s.v[j];
                    f[j][n] = s.f[j];
                    fdot[j][n] = s.fdot[j];
                }

                fname[n] = s.name;
#ifdef DEBUG
                if (n != int(s.name))
                    fprintf(stderr, "timestep %d: position in array=%d, name=%f\n",
                            timestep, n, s.name);
#endif
                fdummy[n] = s.dummy0;
                t[n] = s.type;
                m[n] = s.mass;
                lm[n] = s.logmass;
                dm[n] = s.diskmass;
                mom[n] = s.momentum;
                ++n;
            }
            break;
            default:
            {
                const Star &s0 = m_stars[i].front();
                const Star &s1 = m_stars[i][1];
                const float dt0 = time - s0.t;
                const float alpha = dt0 / (s1.t - s0.t);
                const float dt1 = time - s1.t;
                for (int j = 0; j < 3; ++j)
                {
#if 1
                    x[j][n] = (s0.x[j] + s0.v[j] * dt0) * (1.f - alpha) + (s1.x[j] + s1.v[j] * dt1) * alpha;
#else
                    x[j][n] = s0.x[j] * (1.f - alpha) + s1.x[j] * alpha;
#endif
                    v[j][n] = s0.v[j] * (1.f - alpha) + s1.v[j] * alpha;
                    f[j][n] = s0.f[j] * (1.f - alpha) + s1.f[j] * alpha;
                    fdot[j][n] = s0.fdot[j] * (1.f - alpha) + s1.fdot[j] * alpha;
                }
                fname[n] = s0.name;
#ifdef DEBUG
                if (n != int(s0.name))
                    fprintf(stderr, "timestep %d: position in array=%d, name=%f\n",
                            timestep, n, s0.name);
#endif
                fdummy[n] = s0.dummy0;
                t[n] = s0.type;
                m[n] = s0.mass * (1.f - alpha) + s1.mass * alpha;
                lm[n] = s0.logmass * (1.f - alpha) + s1.logmass * alpha;
                dm[n] = s0.diskmass * (1.f - alpha) + s1.diskmass * alpha;
                mom[n] = s0.momentum * (1.f - alpha) + s1.momentum * alpha;
                ++n;
            }
            break;
            }
        }
        ++timestep;
        time += m_paramStep->getValue();
        fprintf(stderr, "timestep=%d: size=%d, eof=%d\n", timestep, nstars, int(eof));
    }

    close();

    // Create set objects:
    coDoSet *setPoints = new coDoSet(m_outPoint->getObjName(), points.size(), (coDistributedObject **)&points[0]);
    coDoSet *setTypes = new coDoSet(m_outType->getObjName(), types.size(), (coDistributedObject **)&types[0]);
    coDoSet *setMasses = new coDoSet(m_outMass->getObjName(), masses.size(), (coDistributedObject **)&masses[0]);
    coDoSet *setLogMasses = new coDoSet(m_outLogMass->getObjName(), logmasses.size(), (coDistributedObject **)&logmasses[0]);
    coDoSet *setVelocities = new coDoSet(m_outVelocity->getObjName(), velocities.size(), (coDistributedObject **)&velocities[0]);
    coDoSet *setForces = new coDoSet(m_outForce->getObjName(), forces.size(), (coDistributedObject **)&forces[0]);
    coDoSet *setForceDots = new coDoSet(m_outForceDot->getObjName(), forcedots.size(), (coDistributedObject **)&forcedots[0]);
    coDoSet *setDiskMasses = new coDoSet(m_outDiskMass->getObjName(), diskmasses.size(), (coDistributedObject **)&diskmasses[0]);
    coDoSet *setDiskMomentums = new coDoSet(m_outDiskMomentum->getObjName(), diskmomentums.size(), (coDistributedObject **)&diskmomentums[0]);
    coDoSet *setNames = new coDoSet(m_outNames->getObjName(), simNames.size(), (coDistributedObject **)&simNames[0]);
    coDoSet *setDummies = new coDoSet(m_outDummies->getObjName(), dummies.size(), (coDistributedObject **)&dummies[0]);
    // Now the arrays can be cleared:
    points.clear();
    masses.clear();
    logmasses.clear();
    velocities.clear();
    forces.clear();
    forcedots.clear();
    diskmasses.clear();
    diskmomentums.clear();

    // Set timestep attribute:
    if (timestep > 1)
    {
        char buf[1024], scalebuf[1024], basebuf[1024], unitbuf[1024];
        snprintf(buf, sizeof(buf), "%d %d", 0, timestep - 1);
        snprintf(scalebuf, sizeof(scalebuf), "%f", m_paramStep->getValue() * 1000.);
        snprintf(basebuf, sizeof(basebuf), "0");
        snprintf(unitbuf, sizeof(unitbuf), "K years");

#define ADDATTRS(codo)                             \
    codo->addAttribute("TIMESTEP", buf);           \
    codo->addAttribute("TIMESTEPBASE", basebuf);   \
    codo->addAttribute("TIMESTEPSCALE", scalebuf); \
    codo->addAttribute("TIMESTEPUNIT", unitbuf);
        ADDATTRS(setPoints);
        ADDATTRS(setTypes);
        ADDATTRS(setMasses);
        ADDATTRS(setLogMasses);
        ADDATTRS(setVelocities);
        ADDATTRS(setForces);
        ADDATTRS(setForceDots);
        ADDATTRS(setNames);
        ADDATTRS(setDummies);
        ADDATTRS(setDiskMasses);
        ADDATTRS(setDiskMomentums);
#undef ADDATTRS
    }

    // Assign sets to output ports:
    m_outPoint->setCurrentObject(setPoints);
    m_outType->setCurrentObject(setTypes);
    m_outMass->setCurrentObject(setMasses);
    m_outLogMass->setCurrentObject(setLogMasses);
    m_outVelocity->setCurrentObject(setVelocities);
    m_outForce->setCurrentObject(setForces);
    m_outForceDot->setCurrentObject(setForceDots);
    m_outNames->setCurrentObject(setNames);
    m_outDummies->setCurrentObject(setDummies);
    m_outDiskMass->setCurrentObject(setDiskMasses);
    m_outDiskMomentum->setCurrentObject(setDiskMomentums);

    fprintf(stderr, "minname=%f, maxname=%f\n", minname, maxname);

    sendInfo("Timesteps loaded: %d (%lu stars)", timestep, static_cast<unsigned long>(m_stars.size()));

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, ReadAstro)
