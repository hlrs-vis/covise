/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * TestRotateInteractionHandler.cpp
 *
 *  Created on: Mar 13, 2012
 *      Author: jw_te
 */

#include "TestRotateInteractionHandler.h"

#include <config/CoviseConfig.h>

#include <osg/Vec3>
#include <math.h>

#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>

#include <cover/VRSceneGraph.h>

namespace TwoHandInteraction
{

ostream &
operator<<(ostream &os, const osg::Matrixf &transform)
{
    os << "[osg::Matrixf: " << endl;
    os << "[ " << transform(0, 0) << ", " << transform(0, 1) << ", "
       << transform(0, 2) << ", " << transform(0, 3) << "]" << endl;
    os << "[ " << transform(1, 0) << ", " << transform(1, 1) << ", "
       << transform(1, 2) << ", " << transform(1, 3) << "]" << endl;
    os << "[ " << transform(2, 0) << ", " << transform(2, 1) << ", "
       << transform(2, 2) << ", " << transform(2, 3) << "]" << endl;
    os << "[ " << transform(3, 0) << ", " << transform(3, 1) << ", "
       << transform(3, 2) << ", " << transform(3, 3) << "]" << endl;
    os << "]" << endl;
    return os;
}

TwoHandInteractionPlugin::InteractionResult
TestRotateInteractionHandler::CalculateInteraction(double frameTime,
                                                   const TwoHandInteractionPlugin::InteractionStart &interactionStart,
                                                   bool buttonPressed, const osg::Matrix &handMatrix,
                                                   const osg::Matrix &secondHandMatrix)
{
    TwoHandInteractionPlugin::InteractionResult result(interactionStart);

    // we want rotation and translation combined
    orig_transrotm = interactionStart.RotationMatrix * interactionStart.TranslationMatrix;
    result.RotationMatrix.set(orig_transrotm);
    result.TranslationMatrix.set(osg::Matrix());
    // get scale
    orig_scalem = interactionStart.ScalingMatrix;

    // calculate hand
    handRight = handMatrix.getTrans();
    handLeft = secondHandMatrix.getTrans();
    delta = handLeft - handRight;
    handmat.makeTranslate((handRight + handLeft) / 2);

    bool doTransRot = 1;
    bool doScale = 1;

    if (buttonPressed)
    {

        if (!buttonWasPressed)
        {
            old_delta = delta;
            old_handmat = handmat;
            old_handmatinvert = osg::Matrix::inverse(old_handmat);

            old_transrotm = orig_transrotm;
            old_scalem = orig_scalem;
        }
        else
        {
            // Hand rotation is always relative to the startOrientation.
            // Thats why we can calculate the rotation only while the button is pressed.
            // The resulting handmat contains the translation and a rotation around the translated point.
            osg::Matrix handrotm;
            handrotm.makeRotate(old_delta, delta);
            handmat = handrotm * handmat;

            if (doTransRot)
            {
                osg::Matrix relMat;
                relMat = old_handmatinvert * handmat;

                osg::Matrix new_transrotm;
                new_transrotm = old_transrotm * relMat;

                result.RotationMatrix.set(new_transrotm);
            }

            if (doScale)
            {
                double scaleFactor;
                scaleFactor = (delta.length()) / (old_delta.length());

                // move transrot matrix to keep click position fixed
                // (basically copied from coVRNavigationInteraction::doScale)
                osg::Vec3 delta2 = result.RotationMatrix.getTrans();
                osg::Vec3 delta = delta2 - handmat.getTrans();
                delta *= scaleFactor;
                delta += handmat.getTrans();
                delta -= delta2;
                osg::Matrix new_transrotm;
                new_transrotm = result.RotationMatrix * osg::Matrix::translate(delta[0], delta[1], delta[2]);
                result.RotationMatrix.set(new_transrotm);

                osg::Matrix new_scalem;
                new_scalem = old_scalem * osg::Matrix::scale(scaleFactor, scaleFactor, scaleFactor);
                result.ScalingMatrix.set(new_scalem);
            }
        }
    }

    buttonWasPressed = buttonPressed;
    return result;
}
}

/*
 if(buttonPressed&&delta.length()<lengththreshold&&!thresholdreached)
 {
 thresholdreached=true;
 olddelta=delta;
 }
 else if(buttonPressed&&delta.length()>lengththreshold&&thresholdreached)
 {
 thresholdreached=false;
 }


 if(buttonPressed){

 if(!buttonWasPressed && buttonPressed)
 {

 oldhR = handRight;
 oldhL = handLeft;
 olddelta=delta;

 oldhandRight = handMatrix.getTrans();
 oldhandLeft = secondHandMatrix.getTrans();

 oldm = result.RotationMatrix * result.TranslationMatrix;

 rotm = result.RotationMatrix;
 trans = result.TranslationMatrix.getTrans();
 //scale = result.ScalingMatrix.getScale();
 scalem = result.ScalingMatrix;

 //cerr << "click" << endl;
 }

 if(delta.length()>lengththreshold&&doTrans)
 {
 trans = trans + (handRight-oldhR);
 result.TranslationMatrix.setTrans(trans);
 oldhR = handRight;
 oldhL = handLeft;
 }

 else
 {


 newrotm.makeRotate(olddelta,delta);


 trans = trans + ((handRight-oldhR)+(handLeft-oldhL))/2;
 double deltal = ((delta.length() - olddelta.length()));

 deltal= delta.length()/olddelta.length();

 osg::Vec3 newscale;
 newscale = osg::Vec3(deltal,deltal,deltal);
 osg::Matrix newscalem;
 newscalem.makeScale(newscale);

 osg::Matrix handtransm;
 osg::Matrix inversehandtransm;
 osg::Vec3 handtrans;
 handtrans = (handRight + handLeft)/2;
 //handtrans = transm * result.ScalingMatrix * handtrans;

 osg::Matrix old_mat;


 old_mat.makeTranslate((oldhandRight + oldhandLeft)/2);
 handtransm.makeTranslate(handtrans);

 osg::Matrix rotransm;

 rotransm = newrotm * handtransm;



 osg::Matrix rel_mat, dcs_mat;

 rel_mat.mult(old_mat, rotransm);                //erste handMat * aktualisierte handMat
 dcs_mat.mult(oldm, rel_mat);

 rotm=dcs_mat;
 transm=osg::Matrix();

 inversehandtransm.makeTranslate(-handtrans);

 //m_MiddleHandIndicator->setMatrix(handtransm);

 //transm.makeTranslate(trans);
 //inversetransm.makeTranslate(-trans);


 //rotm = rotm * inversehandtransm * newrotm * handtransm;

 //scalem = scalem * transm * handtransm * newscalem * inversehandtransm * inversetransm;


 osg::Vec3 checkScale;
 checkScale = scalem.getScale();

 //cerr << result.ScalingMatrix << endl;

 if(doRot)
 {
 result.RotationMatrix.set(rotm);
 }

 if(doTrans)
 {
 result.TranslationMatrix.set(transm);
 }

 if(checkScale[0]>minscale&&doScale)
 {
 result.ScalingMatrix.set(scalem);
 }
 else
 {
 scale=osg::Vec3(minscale,minscale,minscale);
 }

 oldhR = handRight;
 oldhL = handLeft;

 //olddelta=delta;

 */
