/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/covise_version.h>
#include "alg/coComplexModules.h"
#include "covise/covise_objalg.h"
#include <api/coFeedback.h>
#include <util/coWristWatch.h>
#include <do/coDoGeometry.h>
#include <do/coDoPoints.h>
#include <util/coviseCompat.h>
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>
#include <pthread.h>
#include "Tracer.h"
#ifndef _WIN32
#include <sys/time.h>
#endif
// #include "pthread_errors.h"
//#define _DEBUG_
//#define _DUBUG_
//#define _PROFILE_

#include "Streamlines.h"
#include "Pathlines.h"
#include "Streaklines.h"
#include "PathlinesStat.h"

#include "PointsParser.h"

void
Tracer::postInst()
{
    p_taskType->show();
    p_MaxPoints->show();
    p_trace_len->show();
    p_min_vel->show();
    p_cycles->disable();
    p_control->disable();
    p_timeNewParticles->disable();
    p_randomOffset->show();
    p_cycles->hide();
    p_control->hide();
    p_timeNewParticles->hide();
    p_randomOffset->hide();
// at first only the main thread may modify the list of
// lazy threads and done threads
#ifndef CO_hp1020
    lockMMutex();
#endif
}

float epsilon;
float epsilon_abs;
float grid_tolerance;
float minimal_velocity;
float stepDuration;
int task_type;
int numOfAnimationSteps;
int cycles;
int control;
int newParticles;
float timeNewParticles;
int startStyle;
float divide_cell;
float max_out_of_cell;
int search_level_polygons;
int skip_initial_steps;
float verschiebung[3];
const coDistributedObject *ini_points;
bool randomOffset;
bool randomStartpoint;
int no_start_points;

// compute, quit and worker_function for all
// platforms with the exception of CO_hp1020
#ifndef CO_hp1020
// accesses to doneThreads are synchronised by m_mutex
pthread_mutex_t m_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t done = PTHREAD_COND_INITIALIZER;

Fifo<int> doneThreads;

pthread_t *w_thread;
void *worker_function(void *arg);

// the array w_mutex synchronises accesses to the PTasks
// referred to by read_task_ below...
pthread_mutex_t *w_mutex;
pthread_cond_t *w_go;

void
Tracer::lockMMutex()
{
    // at first only the main thread may modify the list of
    // lazy threads and done threads
    int diagnostics = pthread_mutex_lock(&m_mutex);
    if (diagnostics != 0)
    {
        sendWarning("Could not lock the main mutex.");
    }
}

void
Tracer::startThreads()
{
    int diagnostics = 0;
#ifdef _DUBUG_
    fprintf(stderr, "startThreads ini\n");
#endif
    if (crewSize_ > 0)
    {
        w_thread = new pthread_t[crewSize_];
        w_mutex = new pthread_mutex_t[crewSize_];
        w_go = new pthread_cond_t[crewSize_];
        int i;
        // initialise mutexes and cond. vars.
        for (i = 0; i < crewSize_; ++i)
        {
            diagnostics = pthread_mutex_init(&w_mutex[i], NULL);
            if (diagnostics != 0)
            {
                sendWarning("Could not initialise a mutex.");
            }
            diagnostics = pthread_cond_init(&w_go[i], NULL);
            if (diagnostics != 0)
            {
                sendWarning("Could not initialise a condition variable.");
            }
        }
#ifdef _DUBUG_
        fprintf(stderr, "startThreads after dynamic init\n");
#endif
        // set up a list of lazy threads; initially all workers are lazy
        // here it is important that we have used lockMMutex!!!
        lazyThreads_.clean();
        for (i = 0; i < crewSize_; ++i)
        {
            lazyThreads_.add(i);
        }

        delete[] read_task_;
        read_task_ = new PTask *[crewSize_]; // delete in destructor!!!
#if !defined(__linux__) || defined(__USE_UNIX98)
        pthread_setconcurrency(crewSize_);
#endif
        for (i = 0; i < crewSize_; ++i)
        {
            // read_task_[i] is a pointer to a PTask
            // that is read or written to by the main thread
            // and the i-th worker thread
            // synchronised with w_mutex[i], the worker thread reads
            // from it the task to be done,
            // and the main thread reads from it the status
            // of the tasks and possible results that need gathering.
            // The value of read_task_[i] is only written by the main_thread
            // and ist access is also synchronised with w_mutex[i].
            read_task_[i] = new PTask(i);
            diagnostics = pthread_mutex_lock(&w_mutex[i]);
            if (diagnostics != 0)
            {
                sendWarning("Could not lock a mutex prior to thread creation.");
            }
            diagnostics = pthread_create(&w_thread[i], NULL, worker_function, &read_task_[i]);
            if (diagnostics != 0)
            {
                sendWarning("Could not create a worker thread.");
            }

#ifdef _DUBUG_
            fprintf(stderr, "before waiting loop\n");
#endif
            while (read_task_[i] != NULL)
            {
#ifdef _DUBUG_
                fprintf(stderr, "before waiting\n");
#endif
                pthread_cond_wait(&w_go[i], &w_mutex[i]);
#ifdef _DUBUG_
                fprintf(stderr, "after waiting\n");
#endif
            }
            // now the worker has got its label
            // and is waiting for real work
            diagnostics = pthread_mutex_unlock(&w_mutex[i]);
            if (diagnostics != 0)
            {
                sendWarning("Could not unlock a mutex after thread creation.");
            }
        }
#ifdef _DUBUG_
        fprintf(stderr, "startThreads ends");
#endif
    }
    else
    {
        delete[] read_task_;
        read_task_ = NULL;
    }
}

void Tracer::terminateThreads()
{
    int i, diagnostics;
    for (i = 0; i < crewSize_; ++i)
    {
        diagnostics = pthread_cancel(w_thread[i]);
        if (diagnostics != 0)
        {
            sendWarning("Could not cancel a thread.");
        }
#ifndef __APPLE__
        // on Mac OS 10.3 this won't return
        void *result;
        diagnostics = pthread_join(w_thread[i], &result);
        if (diagnostics != 0)
        {
            sendWarning("Could not join a thread.");
        }
#endif
    }
    delete[] w_thread;
    w_thread = NULL;
    delete[] w_mutex;
    w_mutex = NULL;
    delete[] w_go;
    w_go = NULL;
}

void *worker_function(void *arg)
{
    PTask **p_read_task = (PTask **)arg;
    int label = (*p_read_task)->get_label();
    int diagnostics = 0;

#ifdef _DEBUG_
    fprintf(stderr, "crew %d\n", label);
#endif
    pthread_mutex_lock(&w_mutex[label]);
#ifdef _DEBUG_
    fprintf(stderr, "crew %d locked\n", label);
#endif
    delete (*p_read_task);
    *p_read_task = 0;
    pthread_cond_signal(&w_go[label]);
#ifdef _DEBUG_
    fprintf(stderr, "crew %d signalled\n", label);
#endif

    while (1)
    {
        while (!(*p_read_task) || (*p_read_task)->get_status() != PTask::SERVICED)
        {
#ifdef _DEBUG_
            fprintf(stderr, "crew %d goes to sleep on w_mutex\n", label);
#endif
            pthread_cond_wait(&w_go[label], &w_mutex[label]);
#ifdef _DEBUG_
            fprintf(stderr, "crew %d awakened\n", label);
#endif
        }
#ifdef _DEBUG_
        fprintf(stderr, "crew %d goes to Solve\n", label);
#endif
        // integrate and set status of finished task
        (*p_read_task)->Solve(epsilon, epsilon_abs);
#ifdef _DEBUG_
        fprintf(stderr, "crew %d returns from Solve\n", label);
#endif

        diagnostics = pthread_mutex_lock(&m_mutex);
        if (diagnostics != 0)
        {
            Covise::sendWarning("A worker thread could not lock the main mutex.");
        }
        // sign in the "book" of threads that have finished and signal
        doneThreads.add(label);
        pthread_cond_signal(&done);
        diagnostics = pthread_mutex_unlock(&m_mutex);
#ifdef _DEBUG_
        fprintf(stderr, "crew %d has signaled and unlocked m_mutex\n", label);
#endif
        if (diagnostics != 0)
        {
            Covise::sendWarning("A worker thread could not lock the main mutex.");
        }
    }
    // the thread never gets here
    return 0;
}
#endif

#ifdef _DEBUG_
void
printObjStr(coDistributedObject *grid)
{
    if (grid->isType("SETELE"))
    {
        int no_elems;
        coDoSet *set = dynamic_cast<coDoSet *>(grid);
        coDistributedObject *const *setList = set->getAllElements(&no_elems);
        int elem;
        cout.setf(ios::hex, ios::basefield);
        cout << "********************************" << endl;
        cout << "Set " << grid->getName() << ' ' << grid << endl;
        for (elem = 0; elem < no_elems; ++elem)
        {
            printObjStr(setList[elem]);
        }
        cout << "********************************" << endl;
    }
    else if (grid->isType("UNSGRD"))
    {
        cout << "Unsgrd " << grid->getName() << ' ' << grid << endl;
    }
    else if (grid->isType("POLYGN"))
    {
        cout << "Polygon " << grid->getName() << ' ' << grid << endl;
    }
}
#endif

int Tracer::compute(const char *)
{
// Automatically adapt our Module's title to the species
    if (autoTitle)
    {
        char buf[64];
        sprintf(buf, "Tracer-%s", get_instance());
        setTitle(buf);
    }

    // if fieldIn is connected then dont output a velocity vector,
    // or the Tracer would try to map vector and scalar data to the same outport
    if (p_field->getCurrentObject() && (p_whatout->getValue() == PTask::V_VEC))
    {
        p_whatout->setValue(PTask::V);
    }

#if defined(_PROFILE_)
    coWristWatch _ww;
#endif

    BBoxAdmin_.setSurname();
#ifndef CO_hp1020
    startThreads();
    int diagnostics = 0;
    int label;
#endif
    if (computeGlobals() < 0)
        return FAIL;
    fillWhatOut(); // read output magnitude choice
    // ignore initial time steps

    // time direction
    int td = p_tdirection->getValue();
    if (td == 0)
        td_ = HTask::FORWARDS;
    else if (td == 1)
        td_ = HTask::BACKWARDS;
    else
        td_ = HTask::BOTH;
    // accept backward integration only for streamlines
    if (p_taskType->getValue() > 1)
    {
        td_ = HTask::FORWARDS;
        //      p_tdirection->setValue(1);
    }

    // set up the HTask (this depends directly on the user's choice)
    HTask *theTask = createHTask();
    if (!theTask)
        return FAIL;

    while (!theTask->Finished())
    {
// we process a time step in this loop
// createPTasks: call setTime and create PTasks
#ifdef _DEBUG_
        fprintf(stderr, "main: begin a time step\n");
#endif
        theTask->createPTasks();
        // theTask.ptasks_ after createPTasks is a list of pointer to
        // tasks of the type described in paragraph 1.
        // which we need for this time step, they may depend
        // on the results obtained from previous time steps
        while (!theTask->allPFinished())
        {
            if (crewSize_ > 0)
            {
#ifndef CO_hp1020
                while (theTask->unserviced() && !lazyThreads_.isEmpty())
                {
                    // decide assignation of an unserviced task in theTask.ptasks_
                    // to a lazy thread
                    label = lazyThreads_.extract();
                    diagnostics = pthread_mutex_lock(&w_mutex[label]);
                    if (diagnostics != 0)
                    {
                        sendWarning("Could not lock a mutex prior to assignTask.");
                    }
                    // assign task
                    theTask->assignTask(&read_task_[label], label);
                    pthread_cond_signal(&w_go[label]);
                    diagnostics = pthread_mutex_unlock(&w_mutex[label]);
                    if (diagnostics != 0)
                    {
                        sendWarning("Could not unlock a mutex after assignTask.");
                    }
                }
#ifdef _DEBUG_
                fprintf(stderr, "main: jobs assigned\n");
#endif
                // go to sleep until a thread signals
                while (doneThreads.isEmpty())
                {
#ifdef _DEBUG_
                    fprintf(stderr, "main: wait for workers\n");
#endif
                    pthread_cond_wait(&done, &m_mutex);
#ifdef _DEBUG_
                    fprintf(stderr, "main: a/some worker/s terminated\n");
#endif
                }
                // add items to list of lazy threads for the threads that
                // have signaled their having finished a task or their
                // not being able to continue
                do
                {
                    label = doneThreads.extract();
                    lazyThreads_.add(label);
// gather thread results
#ifdef _DEBUG_
                    fprintf(stderr, "main: try to gather PTask\n");
#endif
                    diagnostics = pthread_mutex_lock(&w_mutex[label]);
                    if (diagnostics != 0)
                    {
                        sendWarning("Could not lock a mutex prior to gatherPTask.");
                    }
                    // do not forget to increase theTask.no_finished_ here!!!!!!!
                    // and make *read_task_[i] = 0, well, this is not necessary,
                    // as long as the workers have set the status to "finished"
                    theTask->gatherPTask(&read_task_[label]);
#ifdef _DEBUG_
                    fprintf(stderr, "main: PTask gathered\n");
#endif
                    diagnostics = pthread_mutex_unlock(&w_mutex[label]);
                    if (diagnostics != 0)
                    {
                        sendWarning("Could not unlock a mutex after gatherPTask.");
                    }
                } while (!doneThreads.isEmpty());
#endif
            }
            else
            {
                theTask->Solve(epsilon, epsilon_abs); // no multithreading
            }
        } // we are done with a time step (streamlines) or with a couple of
// time steps,
// gather results of a time step
#ifdef _DEBUG_
        fprintf(stderr, "main: gatherTimeStep\n");
#endif
        theTask->WarnIfOutOfDomain();
        theTask->gatherTimeStep();
    } // all time steps are done
    theTask->gatherAll(p_line, p_mag);

#ifdef _COMPLEX_MODULE_
    ComplexObject();
#else
    AddInteractionAttributes();
#endif

    delete theTask;
#ifndef CO_hp1020
    terminateThreads();
#endif
#if defined(_PROFILE_)
    sendInfo("stop run: %6.3f s", _ww.elapsed());
#endif

    // Apply color Attribute to uppermost geometry output object
    coDistributedObject *resGeomObj = p_line->getCurrentObject();
    if (resGeomObj)
    {
        resGeomObj->addAttribute("COLOR", p_color->getValue());
        //also add a proper name for interaction plugin
        if (!resGeomObj->getAttribute("OBJECTNAME"))
            resGeomObj->addAttribute("OBJECTNAME", getTitle());
    }

    return SUCCESS;
}

void
Tracer::AddInteractionAttributes()
{
    // attach attribute always to geometry
    coDistributedObject *attachObj = p_line->getCurrentObject();
    if (!attachObj)
        return;

    coFeedback feedback("Tracer");

    feedback.addPara(p_no_startp);
    feedback.addPara(p_startpoint1);
    feedback.addPara(p_startpoint2);
    feedback.addPara(p_direction);
    feedback.addPara(p_verschiebung_);
    feedback.addPara(p_tdirection);
    feedback.addPara(p_whatout);
    feedback.addPara(p_taskType);
    feedback.addPara(p_startStyle);
    feedback.addPara(p_trace_eps);
    feedback.addPara(p_trace_abs);
    feedback.addPara(p_grid_tol);
    feedback.addPara(p_trace_len);
    feedback.addPara(p_min_vel);
    feedback.addPara(p_MaxPoints);
    feedback.addPara(p_stepDuration);
    feedback.addPara(p_cycles);
    feedback.addPara(p_control);
    feedback.addPara(p_newParticles);
    feedback.addPara(p_timeNewParticles);
    feedback.addPara(p_randomOffset);
    feedback.addPara(p_divide_cell);
    feedback.addPara(p_max_out_of_cell);
    feedback.addPara(p_no_threads_w);
    feedback.addPara(p_search_level_polygons_);
    feedback.addPara(p_skip_initial_steps_);

    feedback.addString("0");
    char *t = new char[strlen(getTitle()) + 1];
    strcpy(t, getTitle());

    for (char *c = t + strlen(t); c > t; c--)
    {
        if (*c == '_')
        {
            *c = '\0';
            break;
        }
    }
    char *ud = new char[strlen(t) + 20];
    strcpy(ud, "SYNCGROUP=");
    strcat(ud, t);
    if (strcmp(t, "Tracer") != 0)
    {
        feedback.addString(ud);
    }
    delete[] t;
    delete[] ud;

    if (fbStyle_ == FEED_NEW || fbStyle_ == FEED_BOTH)
    {
        feedback.apply(attachObj);
    }

    if (fbStyle_ == FEED_OLD || fbStyle_ == FEED_BOTH)
    {
        static char interaction[512];
        if (p_startStyle->getValue() == Tracer::SQUARE)
        {
            sprintf(interaction, "P%s\n%s\n%s\n", get_module(),
                    get_instance(),
                    get_host());
        }
        else if (startStyle == Tracer::LINE)
        {
            sprintf(interaction, "T%s\n%s\n%s\n", get_module(),
                    get_instance(),
                    get_host());
        }
#ifndef _COMPLEX_MODULE_
        else if (startStyle == Tracer::CYLINDER)
        {
            sprintf(interaction, "T%s\n%s\n%s\n", get_module(),
                    get_instance(),
                    get_host());
        }
#endif
        attachObj->addAttribute("FEEDBACK", interaction);
    }
}

#ifdef _COMPLEX_MODULE_

void
Tracer::addFeedbackParams(coFeedback &feedback, const char *&oldstyleAttrib)
{
    // Ugly, but working: return pointer to internal field
    static char interaction[512];
    if (p_startStyle->getValue() == Tracer::SQUARE)
    {
        sprintf(interaction, "P%s\n%s\n%s\n", get_module(),
                get_instance(),
                get_host());
    }
    else if (startStyle == Tracer::LINE)
    {
        sprintf(interaction, "T%s\n%s\n%s\n", get_module(),
                get_instance(),
                get_host());
    }
    oldstyleAttrib = interaction;

    feedback.addPara(p_no_startp);
    feedback.addPara(p_startpoint1);
    feedback.addPara(p_startpoint2);
    feedback.addPara(p_direction);
    feedback.addPara(p_verschiebung_);
    feedback.addPara(p_tdirection);
    feedback.addPara(p_whatout);
    feedback.addPara(p_taskType);
    feedback.addPara(p_startStyle);
    feedback.addPara(p_trace_eps);
    feedback.addPara(p_trace_abs);
    feedback.addPara(p_grid_tol);
    feedback.addPara(p_trace_len);
    feedback.addPara(p_min_vel);
    feedback.addPara(p_MaxPoints);
    feedback.addPara(p_stepDuration);
    feedback.addPara(p_cycles);
    feedback.addPara(p_control);
    feedback.addPara(p_newParticles);
    feedback.addPara(p_timeNewParticles);
    feedback.addPara(p_randomOffset);
    feedback.addPara(p_divide_cell);
    feedback.addPara(p_max_out_of_cell);
    feedback.addPara(p_no_threads_w);
    feedback.addPara(p_search_level_polygons_);
    feedback.addPara(p_skip_initial_steps_);
    feedback.addPara(p_radius);
    feedback.addPara(p_free_start_points_);
    const char *string_param_inipoints = p_free_start_points_->getValue();
    if (startStyle != FREE
        || string_param_inipoints == NULL
        || strlen(string_param_inipoints) == 0)
    {
        feedback.addString("0");
    }
    else
    {
        PointsParser IniPoints(string_param_inipoints);
        if (!IniPoints.IsOK())
        {
            sendWarning("Unexpected error when parsing FreeStartPoints");
            feedback.addString("0");
            return;
        }
        int no_initial_points = IniPoints.getNoPoints();
        float *x_ini = NULL;
        float *y_ini = NULL;
        float *z_ini = NULL;
        IniPoints.getPoints(&x_ini, &y_ini, &z_ini);

        char buf[256];
        sprintf(buf, "%d", no_initial_points);

        string ost(buf);

        int point;
        for (point = 0; point < no_initial_points; ++point)
        {
            sprintf(buf, " %g %g %g", x_ini[point], y_ini[point], z_ini[point]);
            ost += buf;
        }
        delete[] x_ini;
        delete[] y_ini;
        delete[] z_ini;
        feedback.addString(ost.c_str());
    }
}

void
Tracer::ComplexObject()
{
    coDistributedObject *filth = p_GeometryOut->getCurrentObject();
    if (filth)
    {
        filth->destroy();
    }
    delete filth;
    p_GeometryOut->setCurrentObject(NULL);
    // we have to generate a coDoGeometry object for
    // p_GeometryOut, we have to distinguish the coDoPoints and
    // coDoLines case
    coDistributedObject *geo = NULL;
    coDistributedObject *norm = NULL;
    coDistributedObject *complexObjectColors = NULL; // colors that the complex object created by itself

    int repeat = 0;
    // lines
    if (p_line->getCurrentObject() && coObjectAlgorithms::containsType<const coDoLines *>(p_line->getCurrentObject()))
    {
        if (p_tube_width->getValue() <= 0.0) // output lines?
        {
            if (p_trailLength->getValue() > 0) // abschneiden?
            {
                geo = ComplexModules::croppedLinesSet(p_line->getCurrentObject(), p_trailLength->getValue());
            }
            else
            {
                geo = p_line->getCurrentObject();
                repeat = 1;
            }
        }
        else // output polygon tubes from the pathlines
        {
            string tubesName = p_GeometryOut->getObjName();
            tubesName += "_Tubes";
            geo = ComplexModules::Tubelines(
                tubesName.c_str(),
                p_line->getCurrentObject(), p_mag->getCurrentObject(),
                p_tube_width->getValue(), p_radius->getValue(), p_trailLength->getValue(), complexObjectType.c_str(),
                &complexObjectColors);
            repeat = 4;
        }
        if (geo)
        {
            geo->incRefCount();
        }
    }
    //points
    else if (p_line->getCurrentObject() && coObjectAlgorithms::containsType<const coDoPoints *>(p_line->getCurrentObject()))
    {
        if (p_taskType->getValue() == MOVING_POINTS)
        {
            string complexObjectName = p_GeometryOut->getObjName();
            string normalsName = p_GeometryOut->getObjName();
            normalsName += "_Normals";

            // fall back to Moving Points (spheres) if vector data was no option because of old maps
            if (complexObjectType == "SPHERE" || p_whatout->getValue() != PTask::V_VEC)
            {
                complexObjectName += "_Spheres";
                geo = ComplexModules::Spheres(complexObjectName.c_str(), p_line->getCurrentObject(), p_radius->getValue(), normalsName.c_str(), &norm);
                repeat = 17;
            }
            else if (complexObjectType == "BAR_MAGNET")
            {
                complexObjectName += "_BarMagnets";
                geo = ComplexModules::BarMagnets(complexObjectName.c_str(), p_line->getCurrentObject(), p_radius->getValue(), p_mag->getCurrentObject(), normalsName.c_str(), &norm, &complexObjectColors);
            }
            else if (complexObjectType == "COMPASS")
            {
                complexObjectName += "_Compasses";
                geo = ComplexModules::Compass(complexObjectName.c_str(), p_line->getCurrentObject(), p_radius->getValue(), p_mag->getCurrentObject(), normalsName.c_str(), &norm, &complexObjectColors);
            }
            else
            {
                complexObjectName += "_Spheres";
                geo = ComplexModules::Spheres(complexObjectName.c_str(), p_line->getCurrentObject(), p_radius->getValue(), normalsName.c_str(), &norm);
                repeat = 17;
            }
        }
    } // no lines and no moving points
    else
    {
        coDistributedObject *start = p_start->getCurrentObject();
        if (start)
        {
            //coDoPoints *t = (coDoPoints *)start;
            start->incRefCount();
            geo = start;
        }
        else
            return;
    }

    // sample geometry?
    string geo_name;
    if (p_SampleGeom_->getCurrentObject() && p_SampleData_->getCurrentObject())
    {
        geo_name = string(p_GeometryOut->getObjName()) + "_geom";
    }
    else
    {
        geo_name = p_GeometryOut->getObjName();
    }
    coDoGeometry *do_geom = new coDoGeometry(geo_name.c_str(), geo);

    const char *interactionAttrib;
    coFeedback feedback("Tracer");

    // create both kinds of feedback argumrents
    addFeedbackParams(feedback, interactionAttrib);

    if (fbStyle_ == FEED_NEW || fbStyle_ == FEED_BOTH)
    {
        feedback.apply(geo);
    }

    if (fbStyle_ == FEED_OLD || fbStyle_ == FEED_BOTH)
    {
        geo->addAttribute("FEEDBACK", interactionAttrib);
    }

    // create normals
    if (norm)
    {
        do_geom->setNormals(PER_VERTEX, norm);
    }

    // create color outpunt
    string color_name = p_GeometryOut->getObjName();
    color_name += "_Color";

    // min and max for colors
    float min = FLT_MAX, max = -FLT_MAX;
    // get min max if no autocolor
    if (!p_autoScale->getValue() && p_ColorMapIn->getCurrentObject() == NULL)
    {
        min = p_minmax->getValue(0);
        max = p_minmax->getValue(1);
        if (max < min)
        {
            std::swap(min, max);
            p_minmax->setValue(0, min);
            p_minmax->setValue(1, max);
        }
        if (p_autoScale->getValue())
        {
            min = FLT_MAX;
            max = -FLT_MAX;
        }
    }
    else if (p_ColorMapIn->getCurrentObject() != NULL)
    {
        coDoColormap *colorMap = (coDoColormap *)(p_ColorMapIn->getCurrentObject());
        min = colorMap->getMin();
        max = colorMap->getMax();
        p_minmax->setValue(0, min);
        p_minmax->setValue(1, max);
    }

    // colormap
    if (complexObjectColors) // use custom colors of complex object?
    {
        // append custom colors to field colors
        if (repeat > 0) // wenn repeat zusaetzlich angegeben, dann erstelle neue kombinierte farbliste
        {
            int numSetEle;
            coDistributedObject *combinedColors;
            coDistributedObject *color;
            if ((p_tube_width->getValue() > 0.0) && (p_trailLength->getValue() > 0))
            {
                // farben fuer abgeschnittene tubelines
                color = ComplexModules::DataTextureLineCropped(color_name, p_mag->getCurrentObject(),
                                                               p_line->getCurrentObject(), p_ColorMapIn->getCurrentObject(),
                                                               false, repeat, p_trailLength->getValue() + 1, min, max);
            }
            else
            {
                color = ComplexModules::DataTexture(color_name, p_mag->getCurrentObject(),
                                                    p_ColorMapIn->getCurrentObject(), false, repeat, &min, &max);
            }
            if (p_autoScale->getValue())
            {
                p_minmax->setValue(0, min);
                p_minmax->setValue(1, max);
            }

            coDistributedObject **outColorList;

            const coDistributedObject *const *setListComplexObjectColors = ((coDoSet *)complexObjectColors)->getAllElements(&numSetEle);
            const coDistributedObject *const *setListColors = ((coDoSet *)color)->getAllElements(&numSetEle);

            outColorList = new coDistributedObject *[numSetEle + 1];
            outColorList[numSetEle] = NULL;

            for (int curSetEle = 0; curSetEle < numSetEle; curSetEle++)
            {
                int numComplexObjectColors = ((coDoRGBA *)setListComplexObjectColors[curSetEle])->getNumPoints();
                int numColors = ((coDoRGBA *)setListColors[curSetEle])->getNumPoints();

                char colorsName[256];
                sprintf(colorsName, "CombinedColors_RGBA_Data_%d", curSetEle);
                outColorList[curSetEle] = new coDoRGBA(colorsName, numComplexObjectColors + numColors);

                float r, g, b, a;
                for (int i = 0; i < numColors; i++)
                {
                    ((coDoRGBA *)setListColors[curSetEle])->getFloatRGBA(i, &r, &g, &b, &a);
                    ((coDoRGBA *)outColorList[curSetEle])->setFloatRGBA(i, r, g, b, a);
                }
                for (int i = 0; i < numComplexObjectColors; i++)
                {
                    ((coDoRGBA *)setListComplexObjectColors[curSetEle])->getFloatRGBA(i, &r, &g, &b, &a);
                    ((coDoRGBA *)outColorList[curSetEle])->setFloatRGBA(numColors + i, r, g, b, a);
                }
            }

            combinedColors = new coDoSet("CombinedColors", outColorList);

            do_geom->setColors(PER_VERTEX, combinedColors);
        }
        else // benutze nur die eigenen farben vom complex object
        {
            do_geom->setColors(PER_VERTEX, complexObjectColors);
        }
    }
    else // no color object
    {

        coDistributedObject *color;
        //if ((p_tube_width->getValue() > 0.0) && (p_trailLength->getValue() > 0)) {
        if (p_trailLength->getValue() > 0)
        {
            color = ComplexModules::DataTextureLineCropped(color_name, p_mag->getCurrentObject(),
                                                           p_line->getCurrentObject(), p_ColorMapIn->getCurrentObject(), false,
                                                           repeat, p_trailLength->getValue() + 1, min, max);
        }
        else
        {
            color = ComplexModules::DataTexture(color_name,
                                                p_mag->getCurrentObject(),
                                                p_ColorMapIn->getCurrentObject(), false, repeat, &min, &max);
        }
        if (p_autoScale->getValue())
        {
            p_minmax->setValue(0, min);
            p_minmax->setValue(1, max);
        }
        if (color)
        {
            do_geom->setColors(PER_VERTEX, color);
        }
    }

    if (!p_SampleGeom_->getCurrentObject() && !p_SampleData_->getCurrentObject())
    {
        p_GeometryOut->setCurrentObject(do_geom);
    }
    else
    {
        coDistributedObject **setList = new coDistributedObject *[3];
        setList[0] = do_geom;
        string creatorModuleName = get_module();
        creatorModuleName += '_';
        creatorModuleName += get_instance();
        do_geom->addAttribute("CREATOR_MODULE_NAME", creatorModuleName.c_str());
        setList[1] = SampleToGeometry(p_SampleGeom_->getCurrentObject(), p_SampleData_->getCurrentObject());

        setList[2] = NULL;
        coDoSet *do_geometries = new coDoSet(p_GeometryOut->getObjName(), setList);
        delete do_geom;
        delete setList[1];
        delete[] setList;
        p_GeometryOut->setCurrentObject(do_geometries);
    }
}

coDoGeometry *
Tracer::SampleToGeometry(const coDistributedObject *grid,
                         const coDistributedObject *data)
{
    string name = p_GeometryOut->getObjName();
    name += "_Sample";
    grid->incRefCount();
    coDoGeometry *do_geo = new coDoGeometry(name.c_str(), grid);

    string creatorModuleName = get_module();
    creatorModuleName += '_';
    creatorModuleName += get_instance();
    do_geo->addAttribute("CREATOR_MODULE_NAME", creatorModuleName.c_str());

    data->incRefCount();
    do_geo->setNormals(PER_VERTEX, data);
    return do_geo;
}
#endif

Tracer::Tracer(int argc, char **argv)
    : coFunctionModule(argc, argv, "Tracer")
#ifndef CO_hp1020
    , read_task_(NULL)
#endif
{
    const char *TimeChoices[] = { "forward", "backward", "both" };
    const char *MagnitudeChoices[] = { "mag", "v_x", "v_y", "v_z", "time", "id", "v" };
    const char *taskTypeChoices[] = { "Streamlines", "Moving Points", "Pathlines", "Streaklines" };

#ifdef _COMPLEX_MODULE_
    const char *startStyleChoices[] = { "line", "plane", "free" };
    // do Auto Titles?
    autoTitleConfigured = coCoviseConfig::isOn("System.AutoName.TracerComp", false);
#else
    const char *startStyleChoices[] = { "line", "plane", "cylinder" };
    // do Auto Titles?
    autoTitleConfigured = coCoviseConfig::isOn("System.AutoName.Tracer", false);
#endif

    complexObjectType = coCoviseConfig::getEntry("Module.Tracer.ComplexObjectType");

    // initially we do what is configured in covise.config - but User may override
    // by setting his own title: done in param()
    autoTitle = autoTitleConfigured;

    // ports
    p_grid = addInputPort("meshIn", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid|Polygons", "input mesh");
    p_velo = addInputPort("dataIn", "Vec3", "input velo.");
    p_ini_points = addInputPort("pointsIn", "Points|UnstructuredGrid|Polygons|TriangleStrips|Lines|Vec3", "input initial points");
    p_ini_points->setRequired(0);
    p_octtrees = addInputPort("octtreesIn", "OctTree|OctTreeP", "input octtrees");
    p_octtrees->setRequired(0);
    p_field = addInputPort("fieldIn", "Float",
                           "input mapped field");
    p_field->setRequired(0);

#ifdef _COMPLEX_MODULE_
    p_ColorMapIn = addInputPort("colorMapIn", "ColorMap", "color map to create geometry");
    p_ColorMapIn->setRequired(0);
    p_SampleGeom_ = addInputPort("SampleGeom", "UniformGrid", "Sample grid");
    p_SampleGeom_->setRequired(0);
    p_SampleData_ = addInputPort("SampleData", "Vec3", "Sample data");
    p_SampleData_->setRequired(0);
    p_GeometryOut = addOutputPort("geometry", "Geometry", "Geometry output");
#endif
    p_line = addOutputPort("lines", "Lines|Points|TriangleStrips", "output geometry");
    p_mag = addOutputPort("dataOut", "Float|Vec3", "output magnitude");
    p_start = addOutputPort("startingPoints", "Points", "real used starting points");
    // parameters
    p_no_startp = addIntSliderParam("no_startp", "Number of starting points");
    p_no_startp->setValue(1, 100, 10);
    p_startpoint1 = addFloatVectorParam("startpoint1", "First start point");
    p_startpoint1->setValue(0.0, 0.0, 0.0);
    p_startpoint2 = addFloatVectorParam("startpoint2", "Last start point");
    p_startpoint2->setValue(1.0, 0.0, 0.0);
    p_direction = addFloatVectorParam("direction", "Square of starting points");
    p_direction->setValue(0.0, 1.0, 0.0);
    p_cyl_axis = addFloatVectorParam("cyl_axis", "axis of starting cylinder");
    p_cyl_axis->setValue(0.0, 0.0, 1.0);
    p_cyl_radius = addFloatParam("cyl_radius", "diameter of starting cylinder");
    p_cyl_radius->setValue(1.0);
    p_cyl_height = addFloatParam("cyl_height", "height of starting cylinder");
    p_cyl_height->setValue(1.0);
    p_cyl_axispoint = addFloatVectorParam("cyl_bottompoint_on_axis", "point on starting cylinder");
    p_cyl_axispoint->setValue(1.0, 0.0, 0.0);
    p_verschiebung_ = addFloatVectorParam("Displacement", "Shift traces");
    p_verschiebung_->setValue(0.0, 0.0, 0.0);
    p_tdirection = addChoiceParam("tdirection", "Forward, backward or both");
    p_tdirection->setValue(3, TimeChoices, 0);
    p_whatout = addChoiceParam("whatout", "mag, v, vx, vy, vz, time or id");
    p_whatout->setValue(7, MagnitudeChoices, 0);
    p_taskType = addChoiceParam("taskType",
                                "Streamlines, moving points or growing lines");
    p_taskType->setValue(4, taskTypeChoices, 1); // Provisional
#ifdef _COMPLEX_MODULE_
    p_startStyle = addChoiceParam("startStyle", "line, square or free");
    p_startStyle->setValue(3, startStyleChoices, 1);
#else
    p_startStyle = addChoiceParam("startStyle", "line, square or cylinder");
    p_startStyle->setValue(3, startStyleChoices, 1);
#endif
    p_trace_eps = addFloatParam("trace_eps", "relative error control");
    p_trace_eps->setValue(0.00001f);
    p_trace_abs = addFloatParam("trace_abs", "absolute error control");
    p_trace_abs->setValue(0.0001f);
    p_grid_tol = addFloatParam("grid_tol", "grid tolerance for UNSGRD or POLYGN");
    p_grid_tol->setValue(0.0001f);
    p_trace_len = addFloatParam("trace_len", "maximum length");
    p_trace_len->setValue(1.0f);
    p_min_vel = addFloatParam("min_vel", "minimal velocity");
    p_min_vel->setValue(1.0e-3f);
    p_MaxPoints = addInt32Param("MaxPoints", "maximum number of points");
    p_MaxPoints->setValue(1000);
    p_stepDuration = addFloatParam("stepDuration", "Step duration if no REALTIME available");
    p_stepDuration->setValue(0.01f);
    p_cycles = addInt32Param("NoCycles", "number of cycles (dynamic data)");
    p_cycles->setValue(1);
    p_control = addBooleanParam("NoInterpolation", "If true, do not interpolate results for animations on static data");
    p_control->setValue(0);
    p_newParticles = addBooleanParam("ThrowNewParticles", "If true, do throw new particles at the same position for dynamic data");
    p_newParticles->setValue(0);
    p_timeNewParticles = addFloatParam("ParticlesReleaseRate", "Frequency at which new particles are released");
    p_timeNewParticles->setValue(0.0);
    p_randomOffset = addBooleanParam("RandomOffset", "If true, Particles are started at a random offset in stationary data fields");
    p_randomOffset->setValue(0);
    p_randomStartpoints = addBooleanParam("RandomStartpoints", "If true, numStartpoints are randomly picked from the overall number of points in an input mesh");
    p_randomStartpoints->setValue(0);
    p_divide_cell = addFloatParam("divideCell", "Step control when out of domain");
    p_divide_cell->setValue(0.125);
    p_max_out_of_cell = addFloatParam("maxOutOfDomain", "Control how far to integrate when out of domain");
    p_max_out_of_cell->setValue(0.25);
    p_no_threads_w = addInt32Param("NoWThreads", "number of worker threads");
    p_no_threads_w->setValue(crewSize_ = findCrewSize());
    p_search_level_polygons_ = addInt32Param("SearchLevel", "search level for polygons");
    p_search_level_polygons_->setValue(0);
    p_skip_initial_steps_ = addInt32Param("SkipInitialSteps", "skip initial steps");
    p_skip_initial_steps_->setValue(0);
    p_color = addStringParam("color", "attribute color");
    p_color->setValue("red");

    gridName_ = "";
    veloName_ = "";
    iniPName_ = "";
    octTreeName_ = "";
    fieldName_ = "";
#ifdef _COMPLEX_MODULE_
    p_radius = addFloatParam("SphereRadius", "Radius of output spheres");
    p_radius->setValue(0.2f);
    p_tube_width = addFloatParam("TubeWidth", "Width of Pathline");
    p_tube_width->setValue(0.0);
    p_trailLength = addInt32Param("TrailLength", "Length of a pathline");
    p_trailLength->setValue(0);
    p_free_start_points_ = addStringParam("FreeStartPoints", "Free start points");
    p_free_start_points_->setValue("[0.01, 0.01, 0.01]");

    static float defaultMinmaxValues[2] = { 0.0, 0.0 };
    p_minmax = addFloatVectorParam("MinMax", "Minimum and Maximum value");
    p_minmax->setValue(2, defaultMinmaxValues);
    p_autoScale = addBooleanParam("autoScales", "Automatically adjust Min and Max");
    p_autoScale->setValue(0);
#endif

    fbStyle_ = FEED_NEW;
    std::string fbStyleStr = coCoviseConfig::getEntry("System.FeedbackStyle.Tracer");
    if (!fbStyleStr.empty())
    {
        if (0 == strncasecmp("NONE", fbStyleStr.c_str(), 4))
            fbStyle_ = FEED_NONE;
        if (0 == strncasecmp("OLD", fbStyleStr.c_str(), 3))
            fbStyle_ = FEED_OLD;
        if (0 == strncasecmp("NEW", fbStyleStr.c_str(), 3))
            fbStyle_ = FEED_NEW;
        if (0 == strncasecmp("BOTH", fbStyleStr.c_str(), 4))
            fbStyle_ = FEED_BOTH;
    }

    if (p_taskType->getValue() == MOVING_POINTS)
    {
        if ((complexObjectType == "COMPASS") || (complexObjectType == "BAR_MAGNET"))
        {
            p_whatout->setValue(PTask::V_VEC);
        }
    }
}

bool
Tracer::GoodOctTrees()
{
    if (p_octtrees->getCurrentObject() == NULL)
    {
        return true; // no objects is always OK
    }
    return GoodOctTrees(p_grid->getCurrentObject(), p_octtrees->getCurrentObject());
}

bool
Tracer::GoodOctTrees(const coDistributedObject *grid, const coDistributedObject *otree)
{
    if (grid == NULL || otree == NULL)
    {
        return false;
    }
    if (grid->isType("SETELE"))
    {
        if (!otree->isType("SETELE"))
        {
            return false;
        }
        int no_elems;
        const coDoSet *set = dynamic_cast<const coDoSet *>(grid);
        const coDistributedObject *const *setList = set->getAllElements(&no_elems);
        int no_oelems;
        const coDoSet *oset = dynamic_cast<const coDoSet *>(otree);
        const coDistributedObject *const *osetList = oset->getAllElements(&no_oelems);
        if (no_oelems != no_elems)
        {
            return false;
        }
        int elem;
        for (elem = 0; elem < no_elems; ++elem)
        {
            if (!GoodOctTrees(setList[elem], osetList[elem]))
            {
                return false;
            }
        }
#ifdef _CLEAN_UP_
        for (elem = 0; elem < no_elems; ++elem)
        {
            delete setList[elem];
            delete osetList[elem];
        }
        delete[] setList;
        delete[] osetList;
#endif
        return true;
    }
    else if (grid->isType("UNSGRD"))
    {
        if (otree->isType("OCTREE"))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else if (grid->isType("POLYGN"))
    {
        if (otree->isType("OCTREP"))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return true;
}

HTask *
Tracer::createHTask()
{
    HTask *ret = NULL;
    float *x_ini = NULL, *y_ini = NULL, *z_ini = NULL;
    int no_initial_points = 0;
    no_start_points = p_no_startp->getValue();
    // calculate initial points here
    if (!ini_points) // input object ini_points has priority...
    {
// ... if not present test test first string in p_free_start_points_
#ifdef _COMPLEX_MODULE_
        const char *string_param_inipoints = p_free_start_points_->getValue();
        if (startStyle != FREE
            || string_param_inipoints == NULL
            || strlen(string_param_inipoints) == 0)
        {
#endif
            switch (startStyle)
            {
            case LINE:
                fillLine(&x_ini, &y_ini, &z_ini);
                no_initial_points = p_no_startp->getValue();
                break;
            case SQUARE:
                fillSquare(&x_ini, &y_ini, &z_ini);
                no_initial_points = p_no_startp->getValue();
                break;
#ifndef _COMPLEX_MODULE_
            case CYLINDER:
                fillCylinder(&x_ini, &y_ini, &z_ini);
                no_initial_points = p_no_startp->getValue();
                break;
#endif
            default:
                no_initial_points = 0;
                break;
            }
#ifdef _COMPLEX_MODULE_
        }
        else
        {
            PointsParser IniPoints(string_param_inipoints);
            if (!IniPoints.IsOK())
            {
                sendWarning("Unexpected error when parsing FreeStartPoints");
                return NULL;
            }
            no_initial_points = IniPoints.getNoPoints();
            IniPoints.getPoints(&x_ini, &y_ini, &z_ini);
            p_start->setCurrentObject(new coDoPoints(p_start->getObjName(), no_initial_points, x_ini, y_ini, z_ini));
        }
#endif
    }
    if (p_grid->getCurrentObject()->isType("SETELE") && p_grid->getCurrentObject()->getAttribute("TIMESTEP") && p_velo->getCurrentObject()->isType("SETELE") && !p_velo->getCurrentObject()->getAttribute("TIMESTEP"))
    {
        /* FIXME modifying input objects causes crashes 
      p_velo->getCurrentObject()->addAttribute("TIMESTEP", "1 1");
      */
    }

    switch (p_taskType->getValue())
    {
    case STREAMLINES:
        ret = new Streamlines(this, p_line->getObjName(), p_mag->getObjName(),
                              p_grid->getCurrentObject(), p_velo->getCurrentObject(), ini_points,
                              no_initial_points, x_ini, y_ini, z_ini,
                              p_MaxPoints->getValue(), p_trace_len->getValue(), td_,
                              p_field->getCurrentObject());
        break;
    case MOVING_POINTS:
    case GROWING_LINES:
        // if we have static data, use PathlinesStat and not Pathlines!!!!!
        if (p_grid->getCurrentObject()->isType("SETELE")
            && p_grid->getCurrentObject()->getAttribute("TIMESTEP"))
        {
            ret = new Pathlines(this, p_line->getObjName(), p_mag->getObjName(),
                                p_grid->getCurrentObject(), p_velo->getCurrentObject(), ini_points,
                                no_initial_points, x_ini, y_ini, z_ini, p_field->getCurrentObject());
        }
        else
        {
            ret = new PathlinesStat(this, p_line->getObjName(), p_mag->getObjName(),
                                    p_grid->getCurrentObject(), p_velo->getCurrentObject(), ini_points,
                                    no_initial_points, x_ini, y_ini, z_ini, p_field->getCurrentObject());
        }
        break;
    // these HTasks are still under constructions...
    // or, put it more precisely, nothing has been done yet
    case STREAKLINES:
        if (p_grid->getCurrentObject()->isType("SETELE")
            && p_grid->getCurrentObject()->getAttribute("TIMESTEP"))
        {
            ret = new Streaklines(this, p_line->getObjName(), p_mag->getObjName(),
                                  p_grid->getCurrentObject(), p_velo->getCurrentObject(), ini_points,
                                  no_initial_points, x_ini, y_ini, z_ini, p_field->getCurrentObject());
        }
        else
        {
            sendWarning("For streaklines with static data, please choose the pathlines option and adjust the relevant parameters for that case");
            return NULL;
        }
        break;
    default:
        break;
    }

    int badData = 0;
    if ((gridName_ == "" && veloName_ == "" && iniPName_ == "" && octTreeName_ == "" && fieldName_ == "") // first computation
        // or got different objs
        || gridName_ != p_grid->getCurrentObject()->getName()
        || veloName_ != p_velo->getCurrentObject()->getName()
        || (p_field->getCurrentObject() && // there is an external field and is different
            fieldName_ != p_field->getCurrentObject()->getName())
        || (!p_field->getCurrentObject() && // there is no external field and there was one
            fieldName_ != "")
        || (ini_points && // there are ini_points and are different
            iniPName_ != ini_points->getName())
        || (!ini_points && // there are no ini_points and we used to have
            iniPName_ != "")
        || (p_octtrees->getCurrentObject() && octTreeName_ != p_octtrees->getCurrentObject()->getName())
        || (!p_octtrees->getCurrentObject() && octTreeName_ != ""))
    {
        badData = ret->Diagnose();

        if (badData == 0 && GoodOctTrees())
        {
            gridName_ = p_grid->getCurrentObject()->getName();
            veloName_ = p_velo->getCurrentObject()->getName();
            iniPName_ = (ini_points) ? ini_points->getName() : "";
            fieldName_ = (p_field->getCurrentObject()) ? p_field->getCurrentObject()->getName() : "";
            octTreeName_ = (p_octtrees->getCurrentObject()) ? p_octtrees->getCurrentObject()->getName() : "";
            BBoxAdmin_.load(p_grid->getCurrentObject(), p_octtrees->getCurrentObject());
        }
        else
        {
            gridName_ = "";
            veloName_ = "";
            iniPName_ = "";
            octTreeName_ = "";
            fieldName_ = "";
            delete ret;
            ret = NULL;
            if (!GoodOctTrees())
            {
                sendWarning("Input octtrees structure does not match that of the input grids");
            }
            return NULL;
        }
    }
    else // not first comp. and got the same objects
    {
        BBoxAdmin_.reload(p_grid->getCurrentObject(), p_octtrees->getCurrentObject());
    }

    ret->AssignOctTrees(&BBoxAdmin_);

    return ret;
}

void
Tracer::param(const char *paramname, bool inMapLoading)
{
    // title: If user sets it, we have to de-activate auto-names
    if (strcmp(paramname, "SetModuleTitle") == 0)
    {
        // find out "real" module name
        char realTitle[1024];
        sprintf(realTitle, "%s_%s", get_module(), get_instance());

        // if it differs from the title - disable automatig settings
        if (strcmp(realTitle, getTitle()) != 0)
            autoTitle = false;
        else
            autoTitle = autoTitleConfigured; // otherwise do whatever configured
        return;
    }

    if (strcmp(paramname, p_taskType->getName()) == 0)
    {
        switch (p_taskType->getValue())
        {
        case STREAMLINES:
            p_tdirection->show();
            p_tdirection->enable();
            p_trace_len->show();
            p_trace_len->enable();
            p_min_vel->show();
            p_min_vel->enable();
            p_MaxPoints->show();
            p_MaxPoints->enable();
            p_stepDuration->hide();
            p_stepDuration->disable();
            p_cycles->hide();
            p_cycles->disable();
            p_control->hide();
            p_control->disable();
            p_newParticles->hide();
            p_newParticles->disable();
            p_timeNewParticles->hide();
            p_timeNewParticles->disable();
            p_randomOffset->hide();
            p_randomOffset->disable();
            break;
        case STREAKLINES:
            p_MaxPoints->hide();
            p_MaxPoints->disable();
        case MOVING_POINTS:
            if ((complexObjectType == "COMPASS") || (complexObjectType == "BAR_MAGNET"))
                p_whatout->setValue(PTask::V_VEC);
            p_newParticles->show();
            p_newParticles->enable();
        case GROWING_LINES:
            p_tdirection->hide();
            p_tdirection->disable();
            p_trace_len->hide();
            p_trace_len->disable();
            p_min_vel->hide();
            p_min_vel->disable();
            if (p_taskType->getValue() != STREAKLINES)
            {
                p_MaxPoints->enable();
                p_MaxPoints->show();
            }
            p_stepDuration->show();
            p_stepDuration->enable();
            p_cycles->show();
            p_cycles->enable();
            p_control->show();
            p_control->enable();
            p_newParticles->show();
            p_newParticles->enable();
            if (p_taskType->getValue() == STREAKLINES)
                p_newParticles->setValue(1);
            p_timeNewParticles->show();
            p_timeNewParticles->enable();
            p_randomOffset->show();
            p_randomOffset->enable();
            break;
        default:
            break;
        }
    }
#ifdef _COMPLEX_MODULE_
    else if (strcmp(paramname, p_tube_width->getName()) == 0)
    {
        if (p_tube_width->getValue() > 0.0)
        {
            p_whatout->setValue(PTask::V_VEC);
        }
        else if (p_field->getCurrentObject() && (p_whatout->getValue() == PTask::V_VEC))
        {
            p_whatout->setValue(PTask::V);
        }
    }
    else if (strcmp(paramname, p_whatout->getName()) == 0)
    {
        if (p_whatout->getValue() != PTask::V_VEC)
        {
            p_tube_width->setValue(0.0);
        }
    }
#endif
#ifndef CO_hp1020
    else if (strcmp(paramname, p_no_threads_w->getName()) == 0)
    {
        crewSize_ = p_no_threads_w->getValue();
        if (crewSize_ < 0)
            crewSize_ = 0;
    }
#endif
    else if (strcmp(paramname, p_newParticles->getName()) == 0)
    {
        if (p_newParticles->getValue())
        {
            p_timeNewParticles->show();
            p_timeNewParticles->enable();
            p_randomOffset->show();
            p_randomOffset->enable();
        }
        else
        {
            p_timeNewParticles->hide();
            p_timeNewParticles->disable();
            p_randomOffset->hide();
            p_randomOffset->disable();
        }
    }
    else if (strcmp(paramname, p_startStyle->getName()) == 0)
    {
#ifdef _COMPLEX_MODULE_
        if (p_startStyle->getValue() == FREE) // free points
        {
            p_free_start_points_->show();
            p_free_start_points_->enable();
            p_no_startp->hide();
            p_no_startp->disable();
            p_startpoint1->hide();
            p_startpoint1->disable();
            p_startpoint2->hide();
            p_startpoint2->disable();
            p_direction->hide();
            p_direction->disable();
        }
        else
        {
            p_free_start_points_->hide();
            p_free_start_points_->disable();
            p_no_startp->show();
            p_no_startp->enable();
            p_startpoint1->show();
            p_startpoint1->enable();
            p_startpoint2->show();
            p_startpoint2->enable();
            if (p_startStyle->getValue() == SQUARE)
            {
                p_direction->show();
                p_direction->enable();
            }
            else // line
            {
                p_direction->hide();
                p_direction->disable();
            }
        }
#else
        if (p_startStyle->getValue() == LINE) //line
        {
            p_cyl_axis->hide();
            p_cyl_axis->disable();
            p_cyl_radius->hide();
            p_cyl_radius->disable();
            p_cyl_height->hide();
            p_cyl_height->disable();
            p_cyl_axispoint->hide();
            p_cyl_axispoint->disable();
            p_direction->hide();
            p_direction->disable();
            p_startpoint1->show();
            p_startpoint1->enable();
            p_startpoint2->show();
            p_startpoint2->enable();
        }
        else if (p_startStyle->getValue() == SQUARE) // square
        {
            p_cyl_axis->hide();
            p_cyl_axis->disable();
            p_cyl_radius->hide();
            p_cyl_radius->disable();
            p_cyl_height->hide();
            p_cyl_height->disable();
            p_cyl_axispoint->hide();
            p_cyl_axispoint->disable();
            p_direction->show();
            p_direction->enable();
            p_startpoint1->show();
            p_startpoint1->enable();
            p_startpoint2->show();
            p_startpoint2->enable();
        }
        else // cylinder
        {
            p_cyl_axis->show();
            p_cyl_axis->enable();
            p_cyl_radius->show();
            p_cyl_radius->enable();
            p_cyl_height->show();
            p_cyl_height->enable();
            p_cyl_axispoint->show();
            p_cyl_axispoint->enable();
            p_direction->hide();
            p_direction->disable();
            p_startpoint1->hide();
            p_startpoint1->disable();
            p_startpoint2->hide();
            p_startpoint2->disable();
        }
#endif
    }
#ifdef _COMPLEX_MODULE_
    else if (!inMapLoading && strcmp(paramname, p_minmax->getName()) == 0)
    {
        p_autoScale->setValue(0);
    }
#else
    (void)inMapLoading;
#endif
}

//===============================================================================================
//===============================================================================================
//===============================================================================================
//===============================================================================================

// read parameters
int
Tracer::computeGlobals()
{
    ini_points = extractPoints(p_ini_points->getCurrentObject());
    if (ini_points && ini_points->isType("POINTS"))
    {
        coDoPoints *ipoints = (coDoPoints *)ini_points;
        if (ipoints->getNumPoints() == 2)
        {
            // set starting points parameter
            float *px, *py, *pz;
            ipoints->getAddresses(&px, &py, &pz);
            p_startpoint1->setValue(px[0], py[0], pz[0]);
            p_startpoint2->setValue(px[1], py[1], pz[1]);
            // do not use given ini_points
            ini_points = NULL;
        }
    }
    else
    {
        ini_points = p_ini_points->getCurrentObject();
    }

    epsilon = p_trace_eps->getValue();
    if (epsilon == 0.0)
        epsilon = 0.000001f;
    epsilon_abs = p_trace_abs->getValue();
    grid_tolerance = p_grid_tol->getValue();
    minimal_velocity = p_min_vel->getValue();
    stepDuration = p_stepDuration->getValue();
    task_type = p_taskType->getValue();
    numOfAnimationSteps = p_MaxPoints->getValue();
    cycles = p_cycles->getValue();
    control = p_control->getValue();
    newParticles = p_newParticles->getValue();
    timeNewParticles = p_timeNewParticles->getValue();
    randomOffset = p_randomOffset->getValue();
    randomStartpoint = p_randomStartpoints->getValue();

    startStyle = p_startStyle->getValue();
    divide_cell = p_divide_cell->getValue();
    max_out_of_cell = p_max_out_of_cell->getValue();
    search_level_polygons = p_search_level_polygons_->getValue();
    verschiebung[0] = p_verschiebung_->getValue(0);
    verschiebung[1] = p_verschiebung_->getValue(1);
    verschiebung[2] = p_verschiebung_->getValue(2);

    skip_initial_steps = p_skip_initial_steps_->getValue();

    if (search_level_polygons < 0)
    {
        sendWarning("The search level for polygons should be positive or 0.");
        return -1;
    }
    if (epsilon <= 0.0)
    {
        sendWarning("The relative error should be positive.");
        return -1;
    }
    if (epsilon_abs <= 0.0)
    {
        sendWarning("The absolute error should be positive.");
        return -1;
    }
    if (grid_tolerance < 0.0)
    {
        sendWarning("The grid tolerance may not be negative.");
        return -1;
    }
    if (task_type == 1 && minimal_velocity <= 0.0)
    {
        sendWarning("The minimal velocity for streamlines has to be positive.");
        return -1;
    }
    if (numOfAnimationSteps < 0)
    {
        sendWarning("Negative values for MaxPoints are not accepted.");
        return -1;
    }
    if (divide_cell <= 0.0
        || divide_cell >= 0.5)
    {
        sendWarning("divide_cell parameter should lie between 0 and 0.5.");
        return -1;
    }
    if (max_out_of_cell <= divide_cell)
    {
        sendWarning("max_out_of_cell should be larger than divide_cell.");
        return -1;
    }
    if (timeNewParticles < 0.0)
    {
        sendWarning("Frequency at which new partcles are released may not be negative");
        return -1;
    }
    if (cycles <= 0)
    {
        sendWarning("Only positive values for the number of cycles are meaningful.");
        return -1;
    }
    return 0;
}

MODULE_MAIN(Tracer, Tracer)
