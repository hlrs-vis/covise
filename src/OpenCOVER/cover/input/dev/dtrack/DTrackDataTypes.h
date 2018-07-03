/* DTrackTypes: C header file, A.R.T. GmbH
 *
 * Type definitions used in DTrackSDK
 *
 * Copyright (c) 2007-2017, Advanced Realtime Tracking GmbH
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * Version v2.5.0
 *
 */

#ifndef _ART_DTRACK_DATATYPES_H_
#define _ART_DTRACK_DATATYPES_H_

namespace DTrackSDK_Datatypes {

/**
 * 	\brief	Single marker data (3DOF, double)
 */
typedef struct{
	int id;          //!< id number (starting with 1)
	double quality;  //!< quality (0.0 <= qu <= 1.0; -1 not tracked)
	double loc[3];   //!< location (in mm)
} DTrack_Marker_Type_d;

// -----------------------------------------------------------------------------------------------------

/**
 * 	\brief	Standard body data (6DOF, double)
 *
 *	Currently not tracked bodies get a quality of -1.
 */
typedef struct{
	int id;          //!< id number (starting with 0)
	double quality;  //!< quality (0 <= qu <= 1, no tracking if -1)
	double loc[3];   //!< location (in mm)
	double rot[9];   //!< rotation matrix (column-wise)
	double covref[3];   //!< reference point of covariance
	double cov[36];   //!< 6x6-dimensional covariance matrix for the 6d pose (with 3d location in [mm], 3d euler angles in [rad]).
} DTrack_Body_Type_d;

// -----------------------------------------------------------------------------------------------------

/**
 * 	\brief	inertial body data (6DOF, double)
 */
typedef struct{
	int id;          //!< id number (starting with 0)
	int st;			 //!< state of sensor (no tracking 0 or 1 or 2)
	double error;    //!< error (0 in state 0 and 2, increase in state 1 <=360)
	double loc[3];   //!< location (in mm)
	double rot[9];   //!< rotation matrix (column-wise)
} DTrack_Inertial_Type_d;

// -----------------------------------------------------------------------------------------------------

#define DTRACKSDK_FLYSTICK_MAX_BUTTON    16	 //!< A.R.T. FlyStick data: maximum number of buttons
#define DTRACKSDK_FLYSTICK_MAX_JOYSTICK   8	 //!< A.R.T. FlyStick data: maximum number of joystick values

/**
 * 	\brief	A.R.T. Flystick data (6DOF + buttons, double)
 *
 * 	Currently not tracked bodies get a quality of -1.
 *	Note the maximum number of buttons and joystick values.
 */
typedef struct{
	int id;         //!< id number (starting with 0)
	double quality; //!< quality (0 <= qu <= 1, no tracking if -1)
	int num_button; //!< number of buttons
	int button[DTRACKSDK_FLYSTICK_MAX_BUTTON];  //!< button state (1 pressed, 0 not pressed): 0 front, 1..n-1 right to left
	int num_joystick;  //!< number of joystick values
	double joystick[DTRACKSDK_FLYSTICK_MAX_JOYSTICK];  //!< joystick value (-1 <= joystick <= 1); 0 horizontal, 1 vertical
	double loc[3];  //!< location (in mm)
	double rot[9];  //!< rotation matrix (column-wise)
} DTrack_FlyStick_Type_d;

// -----------------------------------------------------------------------------------------------------

#define DTRACKSDK_MEATOOL_MAX_BUTTON    16  //!< Measurement tool data: maximum number of buttons

/**
 * 	\brief	Measurement tool data (6DOF + buttons, double)
 *
 * 	Currently not tracked bodies get a quality of -1.
 * 	Note the maximum number of buttons.
 */
typedef struct{
	int id;         //!< id number (starting with 0)
	double quality; //!< quality (0 <= qu <= 1, no tracking if -1)
	int num_button; //!< number of buttons
	int button[DTRACKSDK_MEATOOL_MAX_BUTTON];  //!< button state (1 pressed, 0 not pressed): 0 front, 1..n-1 right to left
	double loc[3];  //!< location (in mm)
	double rot[9];  //!< rotation matrix (column-wise)
	double tipradius;   //!< radius of tip if applicable
	double cov[9];  //!< covariance of location (in mm^2)
} DTrack_MeaTool_Type_d;

// -----------------------------------------------------------------------------------------------------

/**
 * 	\brief	Measurement reference data (6DOF, double)
 *
 * 	Currently not tracked bodies get a quality of -1.
 */
typedef struct{
	int id;         //!< id number (starting with 0)
	double quality; //!< quality (0 <= qu <= 1, no tracking if -1)
	double loc[3];  //!< location (in mm)
	double rot[9];  //!< rotation matrix (column-wise)
} DTrack_MeaRef_Type_d;

// -----------------------------------------------------------------------------------------------------

#define DTRACKSDK_HAND_MAX_FINGER    5  //!< Fingertracking hand data: maximum number of fingers

/**
 *	\brief	A.R.T.Fingertracking hand data (6DOF + fingers, double)
 *
 *	Currently not tracked bodies get a quality of -1.
 */
typedef struct{
	int id;         //!< id number (starting with 0)
	double quality; //!< quality (0 <= qu <= 1, no tracking if -1)
	int lr;         //!< left (0) or right (1) hand
	int nfinger;    //!< number of fingers (maximum 5)
	double loc[3];  //!< back of the hand: location (in mm)
	double rot[9];  //!< back of the hand: rotation matrix (column-wise)
	struct{
		double loc[3];           //!< location (in mm)
		double rot[9];           //!< rotation matrix (column-wise)
		double radiustip;        //!< radius of tip
		double lengthphalanx[3]; //!< length of phalanxes; order: outermost, middle, innermost
		double anglephalanx[2];  //!< angle between phalanxes
	} finger[DTRACKSDK_HAND_MAX_FINGER];	//!< order: thumb, index finger, middle finger, ...
} DTrack_Hand_Type_d;

// -----------------------------------------------------------------------------------------------------

#define DTRACKSDK_HUMAN_MAX_JOINTS 200  //!< A.R.T human model type

/**
 * 	\brief	A.R.T human model (max 200 joints (6DOF + angles + rotation, double) + fingertracking)
 *
 * 	Currently not tracked bodies get a quality of -1.
 * 	Note the maximum number of buttons.
 */
typedef struct{
	int id;         //!< id of the human model (starting with 0)
	int num_joints; //!< number of joints of the human model
	struct {
		int id;           //!< id of the joint (starting with 0)
		double quality;   //!< quality of the joint (0 <= qu <= 1, no tracking if -1)
		double loc[3];    //!< location of the joint (in mm)
		double ang[3];    //!< angles in relation to the joint coordinate system
		double rot[9];    //!< rotation matrix of the joint (column-wise) in relation to room coordinaten system
	} joint[DTRACKSDK_HUMAN_MAX_JOINTS]; //!< location and orientation of the joint
} DTrack_Human_Type_d;

}

#endif /* ART_DTRACK_DATATYPES_H_ */
