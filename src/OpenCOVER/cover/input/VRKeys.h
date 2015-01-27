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
 *	File			VRKeys.h 				*
 *									*
 *	Description		defines for setData keywords		*
 *									*
 *	Author			D. Rainer				*
 *									*
 *	Date			20.08.97				*
 *				09.07.98 Performer C++ Interface	*
 *									*
 ************************************************************************/
#ifndef __VR_KEYS_H
#define __VR_KEYS_H

#include <util/common.h>

#define SCENE_SIZE 0
#define SCENE_NODE 1
#define VIEW_MAT 2
#define STEREO_MODE 3
#define NEAR_CLIP 4
#define ADD_SEQ 5
#define ADD_SET 6
#define ADD_GEODE 7
#define PIPE_WINDOW 8
#define TB_MAT 9
#define MASTER_SWITCH 10
#define DELETE_NODE 11
#define HAND_MATRIX 12
#define SCALE_MAT 13
#define FEEDBACK_INFO 14
#define PROCESS_NODE 15
#define HAND_TYPE 16
#define MENU_DCS 17
#define OBJECTS_ROOT 18
#define MENU_ROOT 19
#define ADD_ITEM 20
#define SHOW_BBOX 21
#define HIDE_BBOX 22
#define MENU_MAT 23
#define LIGHTING 24
#define XFORM_ON 25
#define XFORM_OFF 26
#define TOGGLE_XFORM 27
#define TOGGLE_SCALE 28
#define C_FEEDBACK 29
#define T_FEEDBACK 30
#define EXPLODE 31
#define INTERSECTED_MENU_NODE 32
#define ISECT_FLAG 33
#define QUIT_INFO 34
#define WIREFRAME 35
#define SCALE_ON 36
#define SCALE_OFF 37
#define VIEW_POS 38
#define I_FEEDBACK 39
#define INTERSECTED_OBJECT_NODE 40
#define REMOVE_ON 41
#define REMOVE_OFF 42
#define TOGGLE_REMOVE 43
#define REMOVE_FLAG 44
#define REMOVE_NODE 45
#define NODE_TRAV_MASK_false 46
#define UNDO 47
#define TOGGLE_UNDO 48
#define HAND_BUTTON 49
#define LOCK_HAND 50
#define WIREFRAME_ON 51
#define WIREFRAME_OFF 52
#define AUTOWIRE_ON 53
#define AUTOWIRE_OFF 54
#define ADD_AVATAR 55
#define CLEAR_UNDO_LIST 56
#define COORDAXIS_ON 57
#define COORDAXIS_OFF 58
#define FREEZE 59
#define STEADYCAM_ON 60
#define STEADYCAM_OFF 61
#define REMOVE_MENU_DCS 62
#define ROTATOR_ON 63
#define ROTATOR_OFF 64
#define TRANSLATE_MAT 65
#define SCALE_FACTOR 66
#define WINDOW_LIST 67
#define FLY_ON 68
#define FLY_OFF 69
#define GLIDE_ON 70
#define GLIDE_OFF 71
#define VIEWERCOLLIDE_ON 72
#define VIEWERCOLLIDE_OFF 73
#define HANDCOLLIDE_ON 74
#define HANDCOLLIDE_OFF 75
#define RUBBER 76

#define WALK_ON 90
#define WALK_OFF 91

#define FRONT_WIN 51
#define RIGHT_WIN 52
#define LEFT_WIN 53
#define BOTTOM_WIN 54

#define MONO 0
#define STEREO 1

#define HAND_LINE 0
#define HAND_SPHERE 1
#define HAND_PLANE 2
#define HAND_CUBE 3
#define HAND_PYRAMID 4
#define HAND_PROBE 7
#define HAND_ANCHOR 8
#define HAND_FLY_LINE 9
#define HAND_DRIVE 11
#define HAND_WALK 12

#define MY_NONE 0

#define MENU_MAIN 0 /* menu types */
#define MENU_STATIC 1
#define MENU_TEMPORARY 2

/* button action types */
#define BUTTON_FUNCTION 0 /* does an action when clicked */
#define BUTTON_SWITCH 1 /* switches between two states */
#define BUTTON_SUBMENU 2 /* invokes a submenu */
#define BUTTON_SLIDER 3 /* float slider */

#define BUTTON_PRESSED 0
#define BUTTON_DRAGGED 1
#define BUTTON_RELEASED 2

#define RUBBER_BAND 0
#define RUBBER_SPHERE 1
#define RUBBER_BOX 2
#define RUBBER_PLANE 3
#endif
