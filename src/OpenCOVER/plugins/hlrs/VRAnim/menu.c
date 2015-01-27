/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -------------------------------------------------------------------
 *
 *   menu.c:
 *
 *     This is part of the program ANIM. It provides some
 *     subroutines to create pull down menus
 *
 *     Date: Mar96
 *
 * ------------------------------------------------------------------- */

/* ------------------------------------------------------------------- */
/* Standard includes                                                   */
/* ------------------------------------------------------------------- */
#include <GL/gl.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------- */
/* Own includefiles                                                    */
/* ------------------------------------------------------------------- */
#include "anim.h"

/* ------------------------------------------------------------------- */
/* Prototypes                                                          */
/* ------------------------------------------------------------------- */
void animCreateMenus(void);
void animUpdateMenus(void);
void animCallbackPlotMenu(int);
void animCallbackHideMenu(int);
void animCallbackFixMenu(int);
void animCallbackFixtransMenu(int);
void animChangeMiscMenu(int);
void animCreateSomeMenus(void);

extern void animCallbackMenu(int);

/* ------------------------------------------------------------------- */
/* External defined global variables                                   */
/* ------------------------------------------------------------------- */
extern struct geometry geo;
extern struct animation str;
extern struct plotdata dat;
extern struct elgeometry elgeo;
extern struct menuentries menus;
extern struct modes mode;
extern struct flags flag;
extern FILE *outfile;

/* ------------------------------------------------------------------- */
/* Subroutines                                                         */
/* ------------------------------------------------------------------- */
void animCreateMenus(void)
{
    /* Create all menus */

    /* create new submenu: input menu */
    menus.input = glutCreateMenu(animCallbackMenu);
    glutAddMenuEntry("Set of Files", INPUT_SET);
    glutAddMenuEntry("Geometric File", INPUT_GEO);

    animCreateSomeMenus();

    /* set and attach main-menu */
    glutSetMenu(menus.mainmenu);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

/* ------------------------------------------------------------------ */
void animUpdateMenus(void)
/* enable menu items that are only available when model is displayed */
{
    int i;
    char *menuentrytitle = NULL; /* menu entry's title */

    /* if no geometry information nor stripped data exists do nothing */
    if (geo.nfiles == 0)
    {
        return;
    }

    /* Geometry information exists */
    /* set and detach main-menu */
    glutSetMenu(menus.mainmenu);
    glutDetachMenu(GLUT_RIGHT_BUTTON);

    /* destroy main menu */
    glutDestroyMenu(menus.mainmenu);
    menus.mainmenu = 0;

    /* destroy input menu (because plotterdata may exist) */
    glutDestroyMenu(menus.input);
    menus.input = 0;

    /* destroy misc menu */
    if (menus.misc != 0)
    {
        glutDestroyMenu(menus.misc);
        menus.misc = 0;
    }

    /* destroy video menu, if flag.video != 0 */
    if ((menus.video != 0) && (flag.video != 0))
    {
        glutDestroyMenu(menus.video);
        menus.video = 0;
    }

    /* free memory and delete submenus plot */
    if (menus.plot != 0)
    {
        glutDestroyMenu(menus.plot);
        menus.plot = 0;
    }

    /* free memory and delete submenus (hide, fixed, fixtrans) */
    if (menus.hide_file != 0)
    {
        OWN_FREE(menus.hide_file);
        if (menus.hide != 0)
        {
            glutDestroyMenu(menus.hide);
            menus.hide = 0;
        }
    }

    if (menus.fixed_file != 0)
    {
        OWN_FREE(menus.fixed_file);
        if (menus.fixed != 0)
        {
            glutDestroyMenu(menus.fixed);
            menus.fixed = 0;
        }
    }

    if (menus.fixtrans_file != 0)
    {
        OWN_FREE(menus.fixtrans_file);
        if (menus.fixtrans != 0)
        {
            glutDestroyMenu(menus.fixtrans);
            menus.fixtrans = 0;
        }
    }

    /* allocate string for menu entry title */
    OWN_CALLOC(menuentrytitle, char, MAXLENGTH);

    if (dat.ndata)
    {
        /* create submenu plot */
        menus.plot = glutCreateMenu(animCallbackPlotMenu);

        /* create submenu: plot menu */
        glutSetMenu(menus.plot);
        for (i = 0; i < dat.ndata; i++)
        {
            /* define menu entry title */
            sprintf(menuentrytitle, "Plot %s", dat.name[i]);
            glutAddMenuEntry(menuentrytitle, i);
        }
        /* set first menu item string to Plotted */
        sprintf(menuentrytitle, "Plotted %s", dat.name[0]);
        glutChangeToMenuEntry(0 + 1, menuentrytitle, 0);
        mode.plotsel = 1;
    }

    /* allocate memory for flags */
    OWN_CALLOC(menus.hide_file, int, (geo.nfiles + elgeo.nfiles));
    OWN_CALLOC(menus.fixed_file, int, geo.nfiles);
    OWN_CALLOC(menus.fixtrans_file, int, geo.nfiles);

    /* create submenus (hide, fixed, fixtrans) */
    menus.hide = glutCreateMenu(animCallbackHideMenu);
    menus.fixed = glutCreateMenu(animCallbackFixMenu);
    menus.fixtrans = glutCreateMenu(animCallbackFixtransMenu);

    for (i = 0; i < geo.nfiles; i++)
    {

        glutSetMenu(menus.fixed);
        /* define menu entry title */
        sprintf(menuentrytitle, "Fix %s", geo.name[i]);
        glutAddMenuEntry(menuentrytitle, i);
        menus.fixed_file[i] = GL_FALSE;

        glutSetMenu(menus.hide);
        /* define menu entry title */
        sprintf(menuentrytitle, "Hide %s", geo.name[i]);
        glutAddMenuEntry(menuentrytitle, i);
        menus.hide_file[i] = GL_FALSE;

        glutSetMenu(menus.fixtrans);
        /* define menu entry title */
        sprintf(menuentrytitle, "Fix %s", geo.name[i]);
        glutAddMenuEntry(menuentrytitle, i);
        menus.fixtrans_file[i] = GL_FALSE;
    }

    for (i = geo.nfiles; i < (geo.nfiles + elgeo.nfiles); i++)
    {
        glutSetMenu(menus.hide);
        /* define menu entry title */
        sprintf(menuentrytitle, "Hide %s", elgeo.name[i - geo.nfiles]);
        glutAddMenuEntry(menuentrytitle, i);
        menus.hide_file[i] = GL_FALSE;
    }

    /* free string for menu entry title */
    OWN_FREE(menuentrytitle);

    glutSetMenu(menus.hide);
    glutAddMenuEntry("Hide All", (geo.nfiles + elgeo.nfiles));

    /* Check if stripped data was read */
    if (str.timesteps == 0)
    {
        /* stripped data doesn't exist */
        /* create submenu: input menu */
        menus.input = glutCreateMenu(animCallbackMenu);
        glutAddMenuEntry("Set of Files", INPUT_SET);
        glutAddMenuEntry("Geometric File", INPUT_GEO);
        glutAddMenuEntry("Stripped File", INPUT_STR);
        glutAddMenuEntry("Colormap File", INPUT_CMP);
        glutAddMenuEntry("Sensor File", INPUT_SNS);
        glutAddMenuEntry("Transformation File", INPUT_TRMAT);
        glutAddMenuEntry("Light File", INPUT_LIG);

        /* create main menu */
        menus.mainmenu = glutCreateMenu(animCallbackMenu);
        glutAddMenuEntry("Exit Anim (q)", ANIM_EXIT);
        glutAddSubMenu("Input", menus.input);
    }
    else
    {
        /* stripped data exists */

        /* create special subsubmenus: Video Menu, MultiImage Menu */

        /* Videomode: "ON" */
        if (flag.video == 0)
        {
            menus.video = glutCreateMenu(animCallbackMenu);
            glutAddMenuEntry("ON", VIDEO_ON);
        }

        /* Videomode: "OFF" */
        if (flag.video == 1)
        {
            menus.video = glutCreateMenu(animCallbackMenu);
            glutAddMenuEntry("Off", VIDEO_OFF);
        }

        /* Videomode: "ON/MPEG" */
        if (flag.video == 2)
        {
            menus.video = glutCreateMenu(animCallbackMenu);
            glutAddMenuEntry("On", VIDEO_ON);
            glutAddMenuEntry("Create video", VIDEO_CREATE);
        }

        /* create submenu: Misc menu */
        menus.misc = glutCreateMenu(animCallbackMenu);
        glutAddMenuEntry("Save transformation", SAVE_TRMAT);
        glutAddMenuEntry("Reset view", RESET);
        glutAddSubMenu("Video", menus.video);
        if (str.multimg == GL_FALSE)
        {
            glutAddMenuEntry("Multi-Image On", MULT_ON);
        }
        else
        {
            glutAddMenuEntry("Multi-Image Off", MULT_OFF);
        }
        glutAddMenuEntry("Start simulation", STARTSIM);
        if (dat.ndata != 0)
        {
            glutAddMenuEntry("Move plotter", MVPLOTTER);
        }
        glutAddMenuEntry("Write Interleaf ASCII file", WRITEILEAF);
        if (mode.coord_show_toggle == GL_FALSE)
        {
            glutAddMenuEntry("Show Coordinate Systems (k)", TOGGLE_COORD);
        }
        else
        {
            glutAddMenuEntry("Hide Coordinate Systems (k)", TOGGLE_COORD);
        }
        glutAddMenuEntry("Input Scaling Factor for Coord.systems", INPUT_COORD_SCALING);

        animCreateSomeMenus();
        /* create submenu: input menu */
        menus.input = glutCreateMenu(animCallbackMenu);
        glutAddMenuEntry("Set of Files", INPUT_SET);
        glutAddMenuEntry("Geometric File", INPUT_GEO);
        glutAddMenuEntry("Stripped File", INPUT_STR);
        glutAddMenuEntry("Colormap File", INPUT_CMP);
        glutAddMenuEntry("Sensor File", INPUT_SNS);
        glutAddMenuEntry("Transformation File", INPUT_TRMAT);
        glutAddMenuEntry("Dynamic Color File", INPUT_DYNCOL);
        glutAddMenuEntry("Data File", INPUT_DAT);
        glutAddMenuEntry("Elastic Geometric File", INPUT_ELGEO);
        glutAddMenuEntry("Light File", INPUT_LIG);

        /* create main menu */
        menus.mainmenu = glutCreateMenu(animCallbackMenu);
        glutAddMenuEntry("Exit Anim (q)", ANIM_EXIT);
        glutAddSubMenu("Input", menus.input);
        glutAddSubMenu("Animation", menus.anim);
        glutAddSubMenu("Realtime", menus.time);
        glutAddSubMenu("Projection", menus.proj);
        glutAddSubMenu("MiddleMouseMode", menus.mode);
        glutAddSubMenu("Shading", menus.shade);
        glutAddSubMenu("Misc", menus.misc);
        if (dat.ndata)
        {
            glutAddSubMenu("Plotter", menus.plot);
        }
        glutAddSubMenu("Hide/Show Objects", menus.hide);
        glutAddSubMenu("Fix/Unfix Observer to Obj.", menus.fixed);
        glutAddSubMenu("Fix/Unfix Obj.'s Trans.", menus.fixtrans);

    } /* if(str.timesteps!=0) */

    /* set and attach main-menu */
    glutSetMenu(menus.mainmenu);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

/* ------------------------------------------------------------------ */
void animCallbackPlotMenu(int value)
{
    char *menuentrytitle = NULL;
    int i;

    /* allocate string for menu entry title */
    OWN_CALLOC(menuentrytitle, char, MAXLENGTH);

    /* set all menu item strings to Plot */
    glutSetMenu(menus.plot);
    for (i = 0; i < dat.ndata; i++)
    {
        /* define menu entry title */
        sprintf(menuentrytitle, "Plot %s", dat.name[i]);
        glutChangeToMenuEntry(i + 1, menuentrytitle, i);
    }

    /* set specified menu item string to Plotted */
    sprintf(menuentrytitle, "Plotted %s", dat.name[value]);
    glutChangeToMenuEntry(value + 1, menuentrytitle, value);
    mode.plotsel = value + 1;

    /* free string for menu entry title */
    OWN_FREE(menuentrytitle);

    /* demand redraw */
    glutPostRedisplay();
}

/* ------------------------------------------------------------------ */
void animCallbackHideMenu(int value)
{
    char *menuentrytitle = NULL; /* menu entry's title */
    static int all_hide = GL_FALSE; /* static hide all flag */
    int i;

    glutSetMenu(menus.hide);

    /* allocate string for menu entry title */
    OWN_CALLOC(menuentrytitle, char, MAXLENGTH);

    if (value < geo.nfiles)
    {
        /* rigid bodies */
        if (menus.hide_file[value] == GL_TRUE)
        {
            /* define menu entry title */
            sprintf(menuentrytitle, "Hide %s", geo.name[value]);
            glutChangeToMenuEntry(value + 1, menuentrytitle, value);
            menus.hide_file[value] = GL_FALSE;
        }
        else
        {
            /* define menu entry title */
            sprintf(menuentrytitle, "Show %s", geo.name[value]);
            glutChangeToMenuEntry(value + 1, menuentrytitle, value);
            menus.hide_file[value] = GL_TRUE;
        }
    }
    else if (value < (geo.nfiles + elgeo.nfiles))
    {
        /* elastic bodies */
        if (menus.hide_file[value] == GL_TRUE)
        {
            /* define menu entry title */
            sprintf(menuentrytitle, "Hide %s", elgeo.name[value - geo.nfiles]);
            glutChangeToMenuEntry(value + 1, menuentrytitle, value);
            menus.hide_file[value] = GL_FALSE;
        }
        else
        {
            /* define menu entry title */
            sprintf(menuentrytitle, "Show %s", elgeo.name[value - geo.nfiles]);
            glutChangeToMenuEntry(value + 1, menuentrytitle, value);
            menus.hide_file[value] = GL_TRUE;
        }
    }
    else
    {
        /* all bodies are chosen */
        if (all_hide == GL_FALSE)
        {
            for (i = 0; i < (geo.nfiles); i++)
            {
                /* define menu entry title */
                sprintf(menuentrytitle, "Show %s", geo.name[i]);
                glutChangeToMenuEntry(i + 1, menuentrytitle, i);
                menus.hide_file[i] = GL_TRUE;
            }

            /* consider elgeofiles with this additional for-loop */
            for (i = geo.nfiles; i < (geo.nfiles + elgeo.nfiles); i++)
            {
                /* define menu entry title */
                sprintf(menuentrytitle, "Show %s", elgeo.name[i - geo.nfiles]);
                glutChangeToMenuEntry(i + 1, menuentrytitle, i);
                menus.hide_file[i] = GL_TRUE;
            }

            /* define menu entry title */
            sprintf(menuentrytitle, "Show all");
            glutChangeToMenuEntry((geo.nfiles + elgeo.nfiles) + 1, menuentrytitle,
                                  (geo.nfiles + elgeo.nfiles));
            all_hide = GL_TRUE;
        }
        else
        {
            for (i = 0; i < (geo.nfiles); i++)
            {
                /* define menu entry title */
                sprintf(menuentrytitle, "Hide %s", geo.name[i]);
                glutChangeToMenuEntry(i + 1, menuentrytitle, i);
                menus.hide_file[i] = GL_FALSE;
            }

            /* consider elgeofiles with this additional for-loop */
            for (i = geo.nfiles; i < (geo.nfiles + elgeo.nfiles); i++)
            {
                /* define menu entry title */
                sprintf(menuentrytitle, "Hide %s", elgeo.name[i - geo.nfiles]);
                glutChangeToMenuEntry(i + 1, menuentrytitle, i);
                menus.hide_file[i] = GL_FALSE;
            }

            /* define menu entry title */
            sprintf(menuentrytitle, "Hide all");
            glutChangeToMenuEntry((geo.nfiles + elgeo.nfiles) + 1, menuentrytitle,
                                  (geo.nfiles + elgeo.nfiles));
            all_hide = GL_FALSE;
        }
    } /* end else */

    /* free string for menu entry title */
    OWN_FREE(menuentrytitle);

    /* demand redraw */
    glutPostRedisplay();
}

/* ------------------------------------------------------------------ */
void animCallbackFixMenu(int value)
{
    char *menuentrytitle = NULL; /* menu entry's title */
    int i;
    int unfixFlag;

    /* allocate string for menu entry title */
    OWN_CALLOC(menuentrytitle, char, MAXLENGTH);

    /* set menus.fixed_file to current menu */
    glutSetMenu(menus.fixed);

    /* test if fixed body was selected */
    unfixFlag = menus.fixed_file[value];

    /* set all menu item strings to fix */
    for (i = 0; i < geo.nfiles; i++)
    {
        /* define menu entry title */
        sprintf(menuentrytitle, "Fix %s", geo.name[i]);
        glutChangeToMenuEntry(i + 1, menuentrytitle, i);
        menus.fixed_file[i] = GL_FALSE;
    }

    /* fix or unfix specified menu item */
    if (unfixFlag == GL_FALSE)
    {
        /* set specified menu item string to unfix */
        sprintf(menuentrytitle, "Unfix %s", geo.name[value]);
        glutChangeToMenuEntry(value + 1, menuentrytitle, value);
        menus.fixed_file[value] = GL_TRUE;
    }

    /* set menus.trans_file to current menu */
    glutSetMenu(menus.fixtrans);

    /* set all menu item strings to fix */
    for (i = 0; i < geo.nfiles; i++)
    {
        /* define menu entry title */
        sprintf(menuentrytitle, "Fix %s", geo.name[i]);
        glutChangeToMenuEntry(i + 1, menuentrytitle, i);
        menus.fixtrans_file[i] = GL_FALSE;
    }

    /* free string for menu entry title */
    OWN_FREE(menuentrytitle);

    /* demand redraw */
    glutPostRedisplay();
}

/* ------------------------------------------------------------------ */
void animCallbackFixtransMenu(int value)
{
    char *menuentrytitle = NULL; /* menu entry's title */
    int i;
    int unfixFlag;

    /* allocate string for menu entry title */
    OWN_CALLOC(menuentrytitle, char, MAXLENGTH);

    /* set menus.trans_file to current menu */
    glutSetMenu(menus.fixtrans);

    /* test if fixed body was selected */
    unfixFlag = menus.fixtrans_file[value];

    /* set all menu item strings to fix */
    for (i = 0; i < geo.nfiles; i++)
    {
        /* define menu entry title */
        sprintf(menuentrytitle, "Fix %s", geo.name[i]);
        glutChangeToMenuEntry(i + 1, menuentrytitle, i);
        menus.fixtrans_file[i] = GL_FALSE;
    }

    /* fix or unfix specified menu item */
    if (unfixFlag == GL_FALSE)
    {
        /* set specified menu item string to unfix */
        sprintf(menuentrytitle, "Unfix %s", geo.name[value]);
        glutChangeToMenuEntry(value + 1, menuentrytitle, value);
        menus.fixtrans_file[value] = GL_TRUE;
    }

    /* set menus.fixed_file to current menu */
    glutSetMenu(menus.fixed);

    /* set all menu item strings to fix */
    for (i = 0; i < geo.nfiles; i++)
    {
        /* define menu entry title */
        sprintf(menuentrytitle, "Fix %s", geo.name[i]);
        glutChangeToMenuEntry(i + 1, menuentrytitle, i);
        menus.fixed_file[i] = GL_FALSE;
    }

    /* free string for menu entry title */
    OWN_FREE(menuentrytitle);

    /* demand redraw */
    glutPostRedisplay();
}

/* ------------------------------------------------------------------ */
void animChangeMiscMenu(int value)
{
    /* add menu entry move to submenu misc */
    glutSetMenu(menus.misc);

    switch (value)
    {

    case MVPLOTTER:
        /* define menu entry title */
        glutChangeToMenuEntry(6, "Move geo", MVGEO);
        mode.move = value;
        break;

    case MVGEO:
        /* define menu entry title */
        glutChangeToMenuEntry(6, "Move plotter", MVPLOTTER);
        mode.move = value;
        break;

    case MULT_ON:
        /* define menu entry title */
        glutChangeToMenuEntry(4, "Multi-Image Off", MULT_OFF);
        break;

    case MULT_OFF:
        /* define menu entry title */
        glutChangeToMenuEntry(4, "Multi-Image On", MULT_ON);
        break;
    }
}

/* ------------------------------------------------------------------ */
void animChangeVideoMenu(int value)
{
    /* add menu entry move to subsubmenu video */
    glutSetMenu(menus.video);

    switch (value)
    {

    case VIDEO_ON:
        /* define menu entry title */
        glutChangeToMenuEntry(1, "Off", VIDEO_OFF);
        if (flag.video != 0)
        {
            glutRemoveMenuItem(2);
        }
        break;

    case VIDEO_OFF:
        /* define menu entry title */
        glutChangeToMenuEntry(1, "On", VIDEO_ON);
        glutAddMenuEntry("Create video", VIDEO_CREATE);
        break;
    }
}

/*---------------------------------------------------------------------*/
void animCreateSomeMenus(void)
{
    /* create subsubmenu: animation->on menu */
    menus.mauto = glutCreateMenu(animCallbackMenu);
    glutAddMenuEntry("Auto (g)", ANIM_AUTO);
    glutAddMenuEntry("Step (d)", ANIM_STEP);

    /* create submenu: animation menu */
    menus.anim = glutCreateMenu(animCallbackMenu);
    glutAddMenuEntry("Off (s)", ANIM_OFF);
    glutAddSubMenu("On", menus.mauto);
    glutAddMenuEntry("Reset (r)", ANIM_RESET);

    /* create submenu: realtime menu */
    menus.time = glutCreateMenu(animCallbackMenu);
    glutAddMenuEntry("Stride", INPUT_STRIDE);
    glutAddMenuEntry("Interval", INPUT_INT);
    glutAddMenuEntry("Calculate Stride", CALC_STRIDE);

    /* create submenu: projection menu */
    menus.proj = glutCreateMenu(animCallbackMenu);
    glutAddMenuEntry("Orthographic", ORTHOGRAPHIC);
    glutAddMenuEntry("Perspectivic", PERSPECTIVE);

    /* create submenu: MiddleMouseMode menu */
    menus.mode = glutCreateMenu(animCallbackMenu);
    glutAddMenuEntry("Zoom (z)", ZOOM);
    glutAddMenuEntry("Translate (t)", TRANSLATE);
    glutAddMenuEntry("Rot-x", ROTX);
    glutAddMenuEntry("Rot-y", ROTY);
    glutAddMenuEntry("Rot-z", ROTZ);

    /* create subsubmenu: Shading->on menu */
    menus.shadem = glutCreateMenu(animCallbackMenu);
    glutAddMenuEntry("Wire (w)", SHADE_WIRE);
    glutAddMenuEntry("Flat", SHADE_FLAT);
    glutAddMenuEntry("Gouraud", SHADE_GOR);

    /* create submenu: Shading menu */
    menus.shade = glutCreateMenu(animCallbackMenu);
    glutAddMenuEntry("Toggle rigid/flex (m)", SHADE_TOGGLE);
    glutAddMenuEntry("Off (o)", SHADE_OFF);
    glutAddSubMenu("On", menus.shadem);

    /* create main-menu */
    menus.mainmenu = glutCreateMenu(animCallbackMenu);
    glutAddMenuEntry("Exit Anim (q)", ANIM_EXIT);
    glutAddSubMenu("Input", menus.input);
}

/*---------------------------------------------------------------------*/
