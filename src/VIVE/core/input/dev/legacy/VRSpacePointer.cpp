/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			VRSpacePointer.C (Performer 2.0)	*
 *									*
 *	Description							*
 *									*
 *	Author			Frank Foehl				*
 *									*
 *	Date			15.1.97					*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include <util/common.h>
#include <cover/coVRPluginSupport.h>

#include "VRSpacePointer.h"
#include "coVRTrackingSystems.h"
#include "VRTracker.h"

using namespace covise;
using namespace opencover;
int gPhantomID = 0;
/************************************************************************/
/*									*/
/* 	VRSpacePointer	class						*/
/*									*/
/************************************************************************/

/*______________________________________________________________________*/
VRSpacePointer::VRSpacePointer()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew VRSpacepointer\n");
#if 0
   mouse.xpos = 0;
   mouse.ypos = 0;
   mouse.wheelcounter = 0;
   mouse.flags = oldflags = 0;
   mouse.modifiers = 0;
#endif
}

/*______________________________________________________________________*/
VRSpacePointer::~VRSpacePointer()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete VRSpacepointer\n");
}

/*______________________________________________________________________*/
void
VRSpacePointer::init(int tt)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nVRSpacePointer::init\n");

    //int st;
    matrix.makeIdentity();
    mx0 = 0.0;
    my0 = 0.0;
    speed = 50.0;
    trackingType = tt;

    if (coVRTrackingUtil::instance()->getTrackingSystem() == coVRTrackingUtil::T_COVER_BEEBOX)
    {

#ifdef BG_LIB
        std::string device;
        device = CoviseConfig::getEntry("COVER.Input.BeeboxConfig.SERIAL_PORT");
        if (device.empty())
        {
            fprintf(stderr, "WARNING: Entry COVER.Input.BeeboxConfig.SERIAL_PORT not found in covise.config Assuming /dev/ttyd2\n");
            device = "/dev/ttyd2";
        }

        bb.debug = 0;
        bb.count = 0;
        bb.n_AChan = 3;
        bb.n_DChan = 8;
        bb.AChan = 0x0b;
        bb.DChan = 0x10;
        bb.Baud = BAUD192;
        bb.StrLen = 2 + (2 * bb.n_AChan) + (bb.n_DChan / 4);
        bb.num_points = 100;

        /*
       ** Open the BG COVER_FLYBOX RS-232 communication.
       */

        if (!open_bgbox(device.c_str(), &bb))
        {
            fprintf(stderr, "Unable to open COVER_BEEBOX port\n");
            exit(2);
        }
        st = init_bgbox(&bb);
        st = w_bg(bb.mode, &bb);
        printf("open_stat %d\n", st);
        r_bg(&bb);
        zero1 = bb.inbuf[0];
        zero2 = bb.inbuf[1];
        zero3 = fb.inbuf[3];
    }
    else if (coVRTrackingUtil::instance()->getTrackingSystem() == coVRTrackingUtil::T_COVER_FLYBOX)
    {
        std::string device;
        device = CoviseConfig::getEntry("COVER.Input.FlyboxConfig.SERIAL_PORT");
        if (device.empty())
        {
            fprintf(stderr, "WARNING: Entry COVER.Input.FlyboxConfig.SERIAL_PORT not found in covise.config Assuming /dev/ttyd2\n");
            device = "/dev/ttyd2";
        }
        fb.debug = 0;
        fb.count = 0;
        fb.n_AChan = 8;
        fb.n_DChan = 16;
        fb.AChan = 0x00;
        fb.AChan |= (AIC1 | AIC2 | AIC3 | AIC4 | AIC5 | AIC6 | AIC7 | AIC8);
        fb.DChan = 0x30;
        fb.Baud = BAUD192;
        fb.StrLen = 2 + (2 * fb.n_AChan) + (fb.n_DChan / 4);
        fb.num_points = 100;

        if (!open_bgbox(device.c_str(), &fb))
        {
            fprintf(stderr, "Unable to open COVER_FLYBOX port\n");
            return;
        }
        st = init_bgbox(&fb);
        printf("open_stat %d\n", st);
        w_bg(fb.mode, &fb);
        r_bg(&fb);
        zero1 = fb.inbuf[0];
        zero2 = fb.inbuf[1];
        zero3 = fb.inbuf[4];
#endif
    }
    else if (coVRTrackingUtil::instance()->getTrackingSystem() == coVRTrackingUtil::T_PHANTOM)
    {
#ifdef PHANTOM_TRACKER
        std::string device;
        device = CoviseConfig::getEntry("COVER.Input.PhantomConfig.SERIAL_PORT");
        if (device.empty())
        {
            fprintf(stderr, "WARNING: Entry COVER.Input.PhantomConfig.SERIAL_PORT not found in covise.config Assuming /dev/ttyd2\n");
            device = "/dev/ttyd2";
        }
        printf("Phantom init: %s\n", device);
        gPhantomID = phid = init_phantom(device);
        printf("phantom_id: %d\n", phid);
        phantom_reset(phid);
        float origin[] = { 0.0, 0.0, 0.0 };
        float nx[] = { 1.0, 0.0, 0.0 };
        float ny[] = { 0.0, 0.0, -1.0 };
        float nz[] = { 0.0, 1.0, 0.0 };
        set_phantom_coordinate_system(phid, origin, nx, ny, nz);
#endif
    }
}

/*______________________________________________________________________*/
void
VRSpacePointer::update(osg::Matrix &mat, unsigned int *button)
{
    (void)mat;

    if (cover->debugLevel(5))
        fprintf(stderr, "\nVRSpacePointer::update\n");

    //float               mx, my;                    //,smooth,tmp_float;
    //pfCoord             coord;
    //float j1,j2,j3=0;

    //matrix wird staendig ueberschrieben =:-)

    if ((coVRTrackingUtil::instance()->getTrackingSystem() == coVRTrackingUtil::T_COVER_FLYBOX) || (trackingType == coVRTrackingUtil::T_COVER_BEEBOX))
    {
#ifdef BG_LIB
        if (coVRTrackingUtil::instance()->getTrackingSystem() == coVRTrackingUtil::T_COVER_BEEBOX)
        {
            r_bg(&bb);

            /*
          **  inbuf[0] - horizontal joy
          **  inbuf[1] - vertical joy
          **  inbuf[3] - bee_slider1
          */

            tmp_float = bb.inbuf[3];
            tmp_float *= 1.25;
            if (tmp_float < -1.0)
            {
                tmp_float = -1.0;
            }
            else if (tmp_float > 1.0)
            {
                tmp_float = 1.0;
            }
            tmp_float += 1.0;
            tmp_float /= 2.0;
            if (tmp_float < 0.0)
            {
                tmp_float = 0.0;
            }
            else if (tmp_float > 1.0)
            {
                tmp_float = 1.0;
            }
            j3 = tmp_float - zero3;
            if ((j3 > -0.1) && (j3 < 0.1))
                j3 = 0;

            //printf( "bee slider = %f\n", tmp_float );

            tmp_float = bb.inbuf[0] - zero1;
            if (tmp_float < 0.0)
            {
                if (tmp_float < -1.0)
                {
                    tmp_float = -1.0;
                }
            }
            else
            {
                if (tmp_float > 1.0)
                {
                    tmp_float = 1.0;
                }
            }
            j1 = tmp_float;

            tmp_float = bb.inbuf[1] - zero2;
            if (tmp_float < 0.0)
            {
                if (tmp_float < -1.0)
                {
                    tmp_float = -1.0;
                }
            }
            else
            {
                if (tmp_float > 1.0)
                {
                    tmp_float = 1.0;
                }
            }
            j2 = tmp_float;

            if (bb.dio & LEFT_MOMENTARY_MASK)
                *button = 1;
            else
                *button = 0;
            st = w_bg(bb.mode, &bb);
        }
        if (coVRTrackingUtil::instance()->getTrackingSystem() == coVRTrackingUtil::T_COVER_FLYBOX)
        {

            r_bg(&fb);

            tmp_float = fb.inbuf[0];
            if (tmp_float < 0.0)
            {
                if (tmp_float < -1.0)
                {
                    tmp_float = -1.0;
                }
            }
            else
            {
                if (tmp_float > 1.0)
                {
                    tmp_float = 1.0;
                }
            }
            j1 = tmp_float - zero1;

            tmp_float = fb.inbuf[1];
            if (tmp_float < 0.0)
            {
                if (tmp_float < -1.0)
                {
                    tmp_float = -1.0;
                }
            }
            else
            {
                if (tmp_float > 1.0)
                {
                    tmp_float = 1.0;
                }
            }
            j2 = tmp_float - zero2;
            tmp_float = fb.inbuf[2];
            if (tmp_float < 0.0)
            {
                if (tmp_float < -1.0)
                {
                    tmp_float = -1.0;
                }
            }
            else
            {
                if (tmp_float > 1.0)
                {
                    tmp_float = 1.0;
                }
            }

            tmp_float = fb.inbuf[4];
            if (tmp_float < 0.0)
            {
                if (tmp_float < -1.0)
                {
                    tmp_float = -1.0;
                }
            }
            else
            {
                if (tmp_float > 1.0)
                {
                    tmp_float = 1.0;
                }
            }
            j3 = tmp_float - zero3;
            if ((j3 > -0.1) && (j3 < 0.1))
                j3 = 0;
            /*
          **  Smooth and set slider 1 value in shared memory.
          */

            fb.inbuf[3] *= 1.25;
            if (fb.inbuf[3] < -1.0)
            {
                fb.inbuf[3] = -1.0;
            }
            else if (fb.inbuf[3] > 1.0)
            {
                fb.inbuf[3] = 1.0;
            }
            fb.inbuf[3] += 1.0;
            fb.inbuf[3] /= 2.0;
            if (fb.inbuf[3] < 0.0)
            {
                fb.inbuf[3] = 0.0;
            }
            else if (fb.inbuf[3] > 1.0)
            {
                fb.inbuf[3] = 1.0;
            }
            smooth = fb.inbuf[3];

            /*
          **  Smooth and set slider 2 value in shared memory.
          */

            fb.inbuf[4] *= 1.25;
            if (fb.inbuf[4] < -1.0)
            {
                fb.inbuf[4] = -1.0;
            }
            else if (fb.inbuf[4] > 1.0)
            {
                fb.inbuf[4] = 1.0;
            }
            smooth = fb.inbuf[4];

/*
          **  Smooth and set pedal 1 value in shared memory.
          */
/*if( !cons4[0] ){
           smooth = fb.inbuf[2] * fb.inbuf[2];
           mused_smooth_pedal( PEDAL1, &smooth );
           f[0] = smooth;
         } else {
           fb.inbuf[5] = -fb.inbuf[5];
           fb.inbuf[5] += 1.0;
           fb.inbuf[5] /= 2.0;
           if( fb.inbuf[5] < 0.0 ){
                fb.inbuf[5] = 0.0;
           } else if ( fb.inbuf[5] > 1.0 ){
         fb.inbuf[5] = 1.0;
         }
         smooth = fb.inbuf[5] * fb.inbuf[5];
         mused_smooth_pedal( PEDAL1, &smooth );
         f[0] = 1.0 - smooth;
         }*/

/*
          **  Get the joystick lever... hooked into analog #8 input
          */
#define THUMB_PIN 7
            if (fb.inbuf[THUMB_PIN] < 0.0)
            {
                smooth = -fb.inbuf[THUMB_PIN] * fb.inbuf[THUMB_PIN];
            }
            else
            {
                smooth = fb.inbuf[THUMB_PIN] * fb.inbuf[THUMB_PIN];
            }
            //f[0] = smooth;

            /*
          **  Set momentary button values in shared memory.
          */

            //if( fb.dio & LEFT_MOMENTARY_MASK )

            //if( fb.dio & RIGHT_MOMENTARY_MASK )

            joy_trigger_value = fb.dio & 0x0100;
            if (joy_trigger_value == JOYSTICK_TRIGGER_MASK)
                *button = 0x04;
            else
                *button = 0;
            /*
          **  Set toggle button values in shared memory.
          */

            /*   if( fb.dio & LEFT_UPPER_TOGGLE_MASK )
                 if( fb.dio & RIGHT_UPPER_TOGGLE_MASK )
                 if( fb.dio & LEFT_MIDDLE_TOGGLE_MASK )
                 if( fb.dio & RIGHT_MIDDLE_TOGGLE_MASK )
                 if( fb.dio & LEFT_LOWER_TOGGLE_MASK )
                 if( fb.dio & RIGHT_LOWER_TOGGLE_MASK )*/
            w_bg(fb.mode, &fb);
        }
        if ((j1 > -0.05) && (j1 < 0.05))
            j1 = 0;
        if ((j2 > -0.05) && (j2 < 0.05))
            j2 = 0;
        if ((j3 > -0.05) && (j3 < 0.05))
            j3 = 0;
        //printf( "j1 = %f\n", j1 );
        //printf( "j2 = %f\n", j2 );
        //printf( "j3 = %f\n", j3 );

        pfGetOrthoMatCoord(matrix, &coord0);
        coord.hpr[0] = coord0.hpr[0] - j1 * speed / 20;
        coord.hpr[1] = coord0.hpr[1] - j2 * speed / 20;
        coord.hpr[2] = 0.0;

        /*if (coord.hpr[1] >= 90.0)
      {
        coord.hpr[0]= -180 + coord.hpr[0];
        coord.hpr[1]= 180.0 - coord.hpr[1];
        coord.hpr[2]= 180.0;
      }*/
        pfMakeCoordMat(matrix, &coord);
        osg::Vec3 v;
        v[0] = 0.0;
        v[1] = j3 * speed / 100;
        v[2] = 0.0;
        pfFullXformPt3(v, v, matrix);
        coord.xyz[0] = coord0.xyz[0] + v[0];
        coord.xyz[1] = coord0.xyz[1] + v[1];
        coord.xyz[2] = coord0.xyz[2] + v[2];
        pfMakeCoordMat(matrix, &coord);
        pfCopyMat(mat, matrix);
#endif
    }
    else if (coVRTrackingUtil::instance()->getTrackingSystem() == coVRTrackingUtil::T_PHANTOM)
    {
#ifdef PHANTOM_TRACKER
        static int oldTime;
        int currTime;
        update_phantom(phid);
        currTime = time(NULL);
        if (get_stylus_switch(phid))
            *button = 1;
        else
        {
            *button = 0;
            oldTime = currTime;
        }
        if (currTime - oldTime > 3)
        {
            oldTime = currTime;
            phantom_reset(gPhantomID);
        }
        get_stylus_matrix(phid, mat.mat);
        mat[3][0] *= 0.2;
        mat[3][1] *= 0.2;
        mat[3][2] *= 0.2;
        osg::Matrix rmat;
        rmat.makeEuler(0.0, -90.0, 0.0);
        mat.preMult(rmat);
#endif
    }
    else if (coVRTrackingUtil::instance()->getTrackingSystem() == coVRTrackingUtil::T_SPACEBALL)
    {

        /* mat.copy(sh->spaceball_mat);
       //fprintf(stderr,"spaceball pos: %f %f %f", mat[3][0], mat[3][1], mat[3][2]);
       *button=0;
       if(sh->button&SPACEBALL_B1)
          *button=1;
       if(sh->button&SPACEBALL_B2)
          *button=DRIVE_BUTTON;
       if(sh->button&SPACEBALL_B3)
          *button=XFORM_BUTTON;
       if(sh->button&SPACEBALL_B5)
          *button=1;
      if(sh->button&SPACEBALL_B6)
      *button=DRIVE_BUTTON;
      if(sh->button&SPACEBALL_B7)
      *button=XFORM_BUTTON;
      if(sh->button&SPACEBALL_BPICK)
      {
      buttonSpecCell spec;
      strcpy(spec.name,"view all");
      sh->button &= ~SPACEBALL_BPICK;
      VRSceneGraph::sg->manipulate(&spec);
      }*/
    }
    else
    {
        //oldflags= mouse.flags; // bugfix dr: coGetMouse wurde in VRRenderer schon aufgerufen
        //coGetMouse(&mouse);

        /* mx= (float)mouse.xpos;
       my= (float)mouse.ypos;

       if (cover->debugLevel(6))
          fprintf(stderr,"mouse x=%f y=%f\n", mx, my);

       matrix.getOrthoCoord(&coord);

       if (mouse.flags & CO_MOUSE_LEFT_DOWN)
          *button= 1;
      else
      *button= 0;

      // clear left mouse button flag
      mouse.flags= mouse.flags & (CO_MOUSE_LEFT_DOWN ^ -1);
      // xy translation
      if ( (mouse.flags & CO_MOUSE_MIDDLE_DOWN) &&
      (mouse.flags & CO_MOUSE_RIGHT_DOWN) )
      {
      if ( ! ((oldflags & CO_MOUSE_MIDDLE_DOWN) &&
      (oldflags & CO_MOUSE_RIGHT_DOWN)) )
      {
      mx0= (float)mouse.xpos;
      my0= (float)mouse.ypos;
      matrix.getOrthoCoord(&coord0);
      }
      coord.xyz[0]= coord0.xyz[0] + (mx-mx0)*speed/1024.0;
      coord.xyz[1]= coord0.xyz[1] + (my-my0)*speed/1024.0;
      }
      // xz translation
      else if (mouse.flags & CO_MOUSE_MIDDLE_DOWN)
      {
      if ( ! ( (oldflags & CO_MOUSE_MIDDLE_DOWN) &&
      !(oldflags & CO_MOUSE_RIGHT_DOWN) &&
      !(oldflags & CO_MOUSE_LEFT_DOWN)) )
      {
      mx0= (float)mouse.xpos;
      my0= (float)mouse.ypos;
      matrix.getOrthoCoord(&coord0);
      }
      coord.xyz[0]= coord0.xyz[0] + (mx-mx0)*speed/1024.0;
      coord.xyz[2]= coord0.xyz[2] + (my-my0)*speed/1024.0;
      }
      //rotation
      else if (mouse.flags & CO_MOUSE_RIGHT_DOWN)
      {
      if ( ! (!(oldflags & CO_MOUSE_MIDDLE_DOWN) &&
      (oldflags & CO_MOUSE_RIGHT_DOWN) &&
      !(oldflags & CO_MOUSE_LEFT_DOWN)) )
      {
      mx0= (float)mouse.xpos;
      my0= (float)mouse.ypos;
      matrix.getOrthoCoord(&coord0);
      }

      coord.hpr[0]= coord0.hpr[0] - (mx-mx0)*speed/256.0;
      coord.hpr[1]= coord0.hpr[1] + (my-my0)*speed/256.0;
      coord.hpr[2]= 0.0;

      if (coord.hpr[1] >= 90.0)
      {
      coord.hpr[0]= -180 + coord.hpr[0];
      coord.hpr[1]= 180.0 - coord.hpr[1];
      coord.hpr[2]= 180.0;
      }
      }

      matrix.makeCoord(&coord);
      mat = matrix;*/
    }

    //oldflags= mouse.flags;

    if (cover->debugLevel(6))
    {
        //mat.print(1,1,"spacepointer mat ",stderr);
        fprintf(stderr, "spacepointer button: %d\n", *button);
    }
}
