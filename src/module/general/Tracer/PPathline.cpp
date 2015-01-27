/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "PPathline.h"
#include "Tracer.h"
#include <float.h>
#include <math.h>
#include <iterator>

// #define _DEBUG_

// returns SERVICED if cell is located, FINISHED_DOMAIN otherwise
PTask::status
PPathline::derivs(float x, // time
                  const float *y, // point
                  float *ydot, // velocity (output)
                  int allGrids,
                  int search_level)
{
    float int_ydot_0[3];
    float int_ydot_1[3];
    status ret_0 = FINISHED_DOMAIN;
    status ret_1 = FINISHED_DOMAIN;
    int i;
    flag_grid_0_ = 1;
    flag_grid_1_ = 1;

    // get "suggestion" for initial grid 0...
    int iniGrid_0 = previousCell_0_.whichGrid();

    if (iniGrid_0 < 0 || iniGrid_0 >= grids0_->size()) // no hint available
    {
        previousCell_0_.notFound();
    } // ...and use it
    else
    {
        ret_0 = derivsForAGrid(y, int_ydot_0, grids0_->operator[](iniGrid_0),
                               vels0_->operator[](iniGrid_0),
                               previousCell_0_.cell_, search_level);
        if (ret_0 == SERVICED)
        {
            flag_grid_0_ = 0;
        }
    }

    if (previousCell_0_.grid_ < 0
        || (ret_0 == FINISHED_DOMAIN
            && emergency_ == GOT_OUT_OF_DOMAIN
            && allGrids))
    {
        for (i = 0; i < grids0_->size(); ++i)
        {
            ret_0 = derivsForAGrid(y, int_ydot_0, grids0_->operator[](i),
                                   vels0_->operator[](i),
                                   previousCell_0_.cell_, search_level);
            if (ret_0 == SERVICED)
            {
                previousCell_0_.setGrid(i);
                flag_grid_0_ = 0;
                break;
            }
        }
    }

    if (ret_0 == FINISHED_DOMAIN)
    {
        // previousCell_0_.notFound();
        return ret_0;
    }

    // get "suggestion" for initial grid 1...
    int iniGrid_1 = previousCell_1_.whichGrid();

    if (iniGrid_1 < 0 || iniGrid_1 >= grids1_->size()) // no hint available
    {
        previousCell_1_.notFound();
    } // ...and use it
    else
    {
        ret_1 = derivsForAGrid(y, int_ydot_1, grids1_->operator[](iniGrid_1),
                               vels1_->operator[](iniGrid_1),
                               previousCell_1_.cell_, search_level);
        if (ret_1 == SERVICED)
        {
            flag_grid_1_ = 0;
        }
    }

    if (previousCell_1_.grid_ < 0
        || (ret_1 == FINISHED_DOMAIN
            && emergency_ == GOT_OUT_OF_DOMAIN
            && allGrids))
    {
        // find interpolation for grids1_ and vels1_
        for (i = 0; i < grids1_->size(); ++i)
        {
            ret_1 = derivsForAGrid(y, int_ydot_1, grids1_->operator[](i),
                                   vels1_->operator[](i),
                                   previousCell_1_.cell_, search_level);
            if (ret_1 == SERVICED)
            {
                previousCell_1_.setGrid(i);
                flag_grid_1_ = 0;
                break;
            }
        }
    }

    if (ret_1 == FINISHED_DOMAIN)
    {
        //      previousCell_1_.notFound();
        return ret_1;
    }

    // interpolate the calculated values (linear interpolation)
    if (realTime_0_ == realTime_1_)
    {
        ydot[0] = int_ydot_0[0];
        ydot[1] = int_ydot_0[1];
        ydot[2] = int_ydot_0[2];
    }
    else
    {
        ydot[0] = int_ydot_0[0] * (realTime_1_ - x);
        ydot[1] = int_ydot_0[1] * (realTime_1_ - x);
        ydot[2] = int_ydot_0[2] * (realTime_1_ - x);
        ydot[0] += int_ydot_1[0] * (x - realTime_0_);
        ydot[1] += int_ydot_1[1] * (x - realTime_0_);
        ydot[2] += int_ydot_1[2] * (x - realTime_0_);
        ydot[0] /= (realTime_1_ - realTime_0_);
        ydot[1] /= (realTime_1_ - realTime_0_);
        ydot[2] /= (realTime_1_ - realTime_0_);
    }
    // cerr << "Time: "<<x<<", z: "<<y[2]<<", zp: "<<ydot[2]<<endl;
    return SERVICED;
}

extern float divide_cell;
extern float max_out_of_cell;
extern int search_level_polygons;

// driver for rkqs stepper
void
PPathline::Solve(float eps, // relative error
                 float eps_abs) // absolute error
{
    int i;
    int kount;
    int nok, nbad;
    int kount_in_domain = 0;
    int kount_out_domain = 0;

    float x, hnext, hdid, h;
    float yscal[3], y[3], dydx[3];
    status problems;
    //   int flag_grid_0_old=0,flag_grid_1_old=0;

    // if keep_status_ is different from NOT_SERVICED,
    // then we have reached a point, beyond which we could
    // not integrate.
    if (keep_status_ != NOT_SERVICED)
    {
        set_status(keep_status_);
        return;
    }

    // get initial conditions
    kount = num_points();
    x = t_c_[kount - 1];
    for (i = 0; i < 3; i++)
    {
        // y[i]=ini_point_[i];
        y[i] = p_c_[i][kount - 1];
    }

    if (kount > 1)
    {
        // h = t_c_[kount-1]-t_c_[kount-2];
        h = last_h_;
    }
    else
    {
        h = realTime_1_ - realTime_0_;
    }
    if (h <= 0.0)
    {
        set_status(keep_status_);
        return;
    }
    nok = nbad = 0;

    // We may not use in general the results from the previous
    // time state...
    // Exceptions: the previous block!!!
    // previousCell_0_.notFound();
    previousCell_0_ = previousCell_1_;
    // previousCell_1_.notFound();

    problems = derivs(x, y, dydx, 0, -1);

    if (problems == FINISHED_DOMAIN)
    {
        previousCell_1_.notFound();
        problems = derivs(x, y, dydx, 0, -1);
    }

    // test initial point for this time interval
    // in ppel it ought to be successful in non-pathological cases.
    if (problems == FINISHED_DOMAIN)
    {
        keep_status_ = FINISHED_DOMAIN;
        set_status(FINISHED_DOMAIN);
        return;
    }
    else
    {
        previousCell_0_.backUp();
        previousCell_1_.backUp();
        set_status(SERVICED);
    }

    while (1)
    {
        // scale for relative error calculation
        for (i = 0; i < 3; i++)
        {
            yscal[i] = fabsf(y[i]) + fabs(dydx[i] * h) + TINY;
        }
        // do not integrate beyond the final point of this time interval
        if ((x + h - realTime_1_) * (x + h - realTime_0_) > 0.0)
        {
            h = realTime_1_ - x;
        }
        // calculate next point (keep input values)
        float x_old, y_old[3];
        x_old = x;
        y_old[0] = y[0];
        y_old[1] = y[1];
        y_old[2] = y[2];
        //gridAndCell debug0S(previousCell_0_);
        //gridAndCell debug1S(previousCell_1_);
        problems = rkqs(y, dydx, 3, &x, h, eps, eps_abs, yscal, &hdid, &hnext);

        if (problems == FINISHED_POINTS)
        {
            set_status(FINISHED_POINTS);
            // in this case we should not try to integrate in future
            // time steps... ?????
            keep_status_ = FINISHED_POINTS;
            return;
        }
        // cerr << "Hdid: "<<hdid<<endl;
        //    if(h==0.0 || hnext==0.0 || hdid==0.0){
        //       cerr <<"Eep"<<endl;
        //    }
        last_h_ = hnext;
        // cerr<<"x: "<<x<<", h: "<<h<<", hdid: "<<hdid<<", hnext: "<<hnext<<endl;

        // test intermediate calculations
        if (emergency_ == OK && problems == FINISHED_DOMAIN)
        {
            // emergency_ = GOT_OUT_OF_DOMAIN;
            kount_in_domain = 0;
            kount_out_domain = 0;
            // restore state of integration (x, y, dydx)?
            // No, rkqs has not modified it in this case.
            // But do restore previousCell_0_... !!!!!!!!!!!!!!!!!!
            previousCell_0_.restore();
            previousCell_1_.restore();
            // Change h control
            float h_ctrl, h_ctrl_0 = FLT_MAX, h_ctrl_1 = FLT_MAX;
            if (flag_grid_0_)
            {
                h_ctrl_0 = suggestInitialH(previousCell_0_.cell_,
                                           grids0_->operator[](previousCell_0_.grid_),
                                           vels0_->operator[](previousCell_0_.grid_));
            }
            if (flag_grid_1_)
            {
                h_ctrl_1 = suggestInitialH(previousCell_1_.cell_,
                                           grids1_->operator[](previousCell_1_.grid_),
                                           vels1_->operator[](previousCell_1_.grid_));
            }
            /*
                  flag_grid_0_old = flag_grid_0_;
                  flag_grid_1_old = flag_grid_1_;
         */
            // find out the minimum
            h_ctrl = h_ctrl_0;
            if (h_ctrl_1 < h_ctrl)
                h_ctrl = h_ctrl_1;
            if (h_ctrl == 0.0)
            {
                set_status(FINISHED_STEP);
                // in this case we should not try to integrate in future
                // time steps... ?????
                keep_status_ = FINISHED_STEP;
                return;
            }
            if (fabs(h) < fabs(h_ctrl))
            {
                emergency_ = GOT_OUT_OF_DOMAIN;
            }
            else
            {
                h = h_ctrl;
            }
            h *= divide_cell;
            continue;
        }
        previousCell_0_.backUp();
        previousCell_1_.backUp();

        // search only in the grids hinted at by previousCell_?_
        problems = derivs(x, y, dydx, 1, -1);

        // test final point
        if (emergency_ == OK && problems == FINISHED_DOMAIN)
        {
            // emergency_ = GOT_OUT_OF_DOMAIN;
            kount_in_domain = 0;
            kount_out_domain = 0;
            // restore state of integration (x, y)?
            // Yes, rkqs has modified it in this case.
            // And do restore previousCell_?_...
            x = x_old;
            y[0] = y_old[0];
            y[1] = y_old[1];
            y[2] = y_old[2];
            previousCell_0_.restore();
            previousCell_1_.restore();
            // Change h control
            float h_ctrl, h_ctrl_0 = FLT_MAX, h_ctrl_1 = FLT_MAX;
            if (flag_grid_0_)
            {
                h_ctrl_0 = suggestInitialH(previousCell_0_.cell_,
                                           grids0_->operator[](previousCell_0_.grid_),
                                           vels0_->operator[](previousCell_0_.grid_));
            }
            if (flag_grid_1_)
            {
                h_ctrl_1 = suggestInitialH(previousCell_1_.cell_,
                                           grids1_->operator[](previousCell_1_.grid_),
                                           vels1_->operator[](previousCell_1_.grid_));
            }
            /*
                  flag_grid_0_old = flag_grid_0_;
                  flag_grid_1_old = flag_grid_1_;
         */
            // find out the minimum
            h_ctrl = h_ctrl_0;
            if (h_ctrl_1 < h_ctrl)
                h_ctrl = h_ctrl_1;
            if (h_ctrl == 0.0)
            {
                set_status(FINISHED_STEP);
                // in this case we should not try to integrate in future
                // time steps... ?????
                keep_status_ = FINISHED_STEP;
                return;
            }
            if (fabs(h) < fabs(h_ctrl))
            {
                emergency_ = GOT_OUT_OF_DOMAIN;
            }
            else
            {
                h = h_ctrl;
            }
            h *= divide_cell;
            continue;
        }
        else if (emergency_ == GOT_OUT_OF_DOMAIN && problems == FINISHED_DOMAIN)
        {
            // the last call to derivs had a 1 as last parameter
            // and we may have overwritten previousCell_0_.cell_
            float cell_size = 0.0;
            if (flag_grid_0_)
            {
                previousCell_0_.restore();
                cell_size = suggestInitialH(previousCell_0_.cell_,
                                            grids0_->operator[](previousCell_0_.grid_),
                                            NULL);
            }
            if (flag_grid_1_)
            {
                previousCell_1_.restore();
                cell_size = suggestInitialH(previousCell_1_.cell_,
                                            grids1_->operator[](previousCell_1_.grid_),
                                            NULL);
            }
            if (cell_size == 0.0)
            {
                cerr << "PPathline: cell_size==0.0" << endl;
            }
            if (getOutLength(y, kount_out_domain)
                > max_out_of_cell * cell_size)
            {
                p_c_[0].resize(p_c_[0].size() - kount_out_domain);
                p_c_[1].resize(p_c_[1].size() - kount_out_domain);
                p_c_[2].resize(p_c_[2].size() - kount_out_domain);
                m_c_.resize(m_c_.size() - kount_out_domain);
                t_c_.resize(t_c_.size() - kount_out_domain);
                u_c_.resize(u_c_.size() - kount_out_domain);
                v_c_.resize(v_c_.size() - kount_out_domain);
                w_c_.resize(w_c_.size() - kount_out_domain);
                hintRegister0_.resize(hintRegister0_.size() - kount_out_domain);
                hintRegister1_.resize(hintRegister1_.size() - kount_out_domain);
                keep_status_ = FINISHED_DOMAIN;
                set_status(FINISHED_DOMAIN);
                break;
            }
            // continue countdown
            ++kount_out_domain;
        }
        else if (emergency_ == GOT_OUT_OF_DOMAIN && problems == SERVICED)
        {
            // have we landed in a new grid?
            if (previousCell_0_.grid_ == previousCell_0_.grid_back_
                && previousCell_1_.grid_ == previousCell_1_.grid_back_)
            {
                // no, the same
                ++kount_in_domain;
                if (kount_in_domain > 4.0 / divide_cell)
                {
                    // seems that the alarm was not correct, so return to normality
                    // kount_in_ok=0;
                    emergency_ = OK;
                }
            }
            else
            {
                // found new grid!!!
                // kount_in_ok=0;
                emergency_ = OK;
                // keep the new info in safety!!!
                previousCell_0_.backUp();
                previousCell_1_.backUp();
                // this might be taken as a good starting point
                // for the new grid, but we may have penetrated it
                // too much. It may be better to check the intermediate
                // results.
                /*
                        status ctrl;
                        for(i=0;i<4;++i){
                           if(intermediate_[i].status_ == FINISHED_DOMAIN){
                              // well, we know this intermediate point was not in the
                              // previous grid, but this does not guarantee that
                              // it is in the actual grid. Let us check this.
                              ctrl=derivs(intermediate_[i].time_,&intermediate_[i].var_[0],
                                     &intermediate_[i].var_dot_[0]);
                              if(ctrl == FINISHED_DOMAIN){
                                 previousCell_0_.restore(); // no, it isn't
            } else {
            x = intermediate_[i].time_;
            y[0] = intermediate_[i].var_[0];
            y[1] = intermediate_[i].var_[1];
            y[2] = intermediate_[i].var_[2];
            dydx[0] = intermediate_[i].var_dot_[0];
            dydx[1] = intermediate_[i].var_dot_[1];
            dydx[2] = intermediate_[i].var_dot_[2];
            previousCell_0_.backUp();
            break;
            }
            }
            }
            */
            }
        }

#ifdef _DEBUG_
        fprintf(stderr, "odeint: eps: %f h: %f hdid: %f\n", eps, h, hdid);
#endif
        addPoint(x, y, dydx, kount, number_);
        hintRegister0_.push_back(previousCell_0_);
        hintRegister1_.push_back(previousCell_1_);
        kount++;
        if (realTime_1_ - x < TINY)
            break;

        if (hdid == h)
            ++nok;
        else
            ++nbad;
        h = hnext;
    }
    if (keep_status_ == NOT_SERVICED)
    {
        set_status(FINISHED_TIME);
    }
}

PPathline::PPathline(float x_ini,
                     float y_ini, float z_ini,
                     std::vector<const coDistributedObject *> &grids0,
                     std::vector<const coDistributedObject *> &vels0,
                     std::vector<const coDistributedObject *> &grids1,
                     std::vector<const coDistributedObject *> &vels1,
                     float time_0,
                     float real_time,
                     int number)
    : PTraceline(x_ini, y_ini, z_ini, grids0, vels0)
{
    release_time_ = real_time;
    grids0_ = &grids0;
    vels0_ = &vels0;
    grids1_ = &grids1;
    vels1_ = &vels1;
    realTime_0_ = time_0;
    realTime_1_ = real_time;
    // look if this point (x_ini, y_ini, z_ini) is in a grid
    float c_ini[3];
    float dydx[3];
    c_ini[0] = x_ini;
    c_ini[1] = y_ini;
    c_ini[2] = z_ini;
    if (derivs(real_time, c_ini, dydx, 0, search_level_polygons) == SERVICED)
    {
        addPoint(real_time, c_ini, dydx, 0, number);
        hintRegister0_.push_back(previousCell_0_);
        hintRegister1_.push_back(previousCell_1_);
        keep_status_ = NOT_SERVICED;
    }
    else
    {
        keep_status_ = FINISHED_DOMAIN;
    }
    last_h_ = 0.0;
}

// see header
void
PPathline::setNewTimeStep(std::vector<const coDistributedObject *> &gridsNew,
                          std::vector<const coDistributedObject *> &velsNew,
                          float x1)
{
    (void)gridsNew;
    (void)velsNew;

    realTime_0_ = realTime_1_;
    realTime_1_ = x1;
}

extern int task_type;

float
PPathline::OutputTime(float integrationTime) const
{
    if (task_type == Tracer::GROWING_LINES)
    {
        return integrationTime;
    }
    return release_time_; // x-release_time_
}

void
PPathline::interpolateField(const vector<const coDistributedObject *> &field,
                            vector<float> &interpolation)
{
    vector<const coDistributedObject *> field0;
    vector<const coDistributedObject *> field1;
    vector<float> l_interpolation;
    unsigned int field_count;
    for (field_count = 0; field_count < field.size() && field[field_count] != NULL;
         ++field_count)
    {
        field0.push_back(field[field_count]);
    }
    if (field[field_count] == NULL && field_count < field.size())
    {
        ++field_count;
        for (; field_count < field.size(); ++field_count)
        {
            field1.push_back(field[field_count]);
        }
    }
    else
    {
        return; // strange... probably a bug
    }
    // field0 and field1 are now correct
    size_t num_interpolations = m_c_.size() - mapped_results_.size();
    for (size_t point = m_c_.size() - num_interpolations;
         point < m_c_.size(); ++point)
    {
        float x = p_c_[0][point];
        float y = p_c_[1][point];
        float z = p_c_[2][point];
        float value;
        float value1 = interpolateFieldInGrid((*grids1_)[hintRegister1_[point].grid_],
                                              field1[hintRegister1_[point].grid_],
                                              hintRegister1_[point].cell_,
                                              x, y, z);
        if ((mapped_results_.size() == 0 && m_c_.size() == 1) || realTime_0_ == realTime_1_)
        {
            value = value1;
        }
        else
        {
            float value0 = interpolateFieldInGrid((*grids0_)[hintRegister0_[point].grid_],
                                                  field0[hintRegister0_[point].grid_],
                                                  hintRegister0_[point].cell_,
                                                  x, y, z);
            float left = (t_c_[point] - realTime_0_) / (realTime_1_ - realTime_0_);
            float right = 1.0f - left;
            value = right * value0 + left * value1;
        }
        l_interpolation.push_back(value);
    }
    l_interpolation.swap(interpolation);
}

float *
PPathline::m_c_interpolate(vector<const coDistributedObject *> &sfield,
                           vector<const coDistributedObject *> &)
{
    vector<float> interpolation;
    if (m_c_.size() == 0)
        return NULL;
    if (mapped_results_.size() < (unsigned int)m_c_.size())
    {
        interpolateField(sfield, interpolation);
        std::copy(interpolation.begin(), interpolation.end(),
                  std::back_inserter(mapped_results_));
    }
    assert(mapped_results_.size() == m_c_.size());
    return const_cast<float *>(&mapped_results_[0]);
}
