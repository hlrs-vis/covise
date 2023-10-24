/*************** <auto-copyright.pl BEGIN do not edit this line> **************
 *
 * osgBullet is (C) Copyright 2009-2012 by Kenneth Mark Bryden
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1 as published by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 *************** <auto-copyright.pl END do not edit this line> ***************/

#include "VrmlMotionState.h"
#include <osgbCollision/Utils.h>
#include <osgbDynamics/TripleBuffer.h>

#include <osgwTools/AbsoluteModelTransform.h>

#include <osg/MatrixTransform>
#include <osg/Notify>
#include <osg/io_utils>



    VrmlMotionState::VrmlMotionState(const osg::Matrix& parentTransform,
        const osg::Vec3& centerOfMass)
        : _parentTransform(parentTransform),
        
        _com(centerOfMass),
        _scale(osg::Vec3(1., 1., 1.)),
        _tb(NULL),
        _tbIndex(0)
    {
        _transform.setIdentity();
    }

    // Sets both the transformation of the collision shape / rigid body in the
    // physics simulation, as well as the matrix of the subgraph's parent
    // Transform node for the OSG visual representation.
    //
    // Called by resetTransform() with the transform (C/S P) -- center of mass
    // over scale, concatenated with the initial parent transform.
    // Apps typically call setCenterOfMass(), setScale(), and
    // setParentTransform() during initialization; these routines then call
    // resetTransform(), which calls here. This results in setting the initial
    // position of both the rigid body and the visual representation.
    //
    // Bullet calls this method during the physics simulation to position
    // collision shapes / rigid bodies.
    //
    // Note that the transformation of the collision shape is not the same as the
    // transformation for the visual representation. VrmlMotionState supports
    // off-origin and scaled visual representations, and thus compensates for
    // differences between scaled and COM-translated collision shapes and
    // unscaled and COM-untranslated OSG subgraphs.
    void
        VrmlMotionState::setWorldTransform(const btTransform& centerOfObjectTransformationInWorldCoordinates)
    {

        const osg::Vec3 cs(_com[0] * _scale[0], _com[1] * _scale[1], _com[2] * _scale[2]);
        const osg::Matrix csMat = osg::Matrix::translate(-cs);
        osg::Matrix parentInverse;
        parentInverse.invert(_parentTransform);
        osg::Matrix localMat =osgbCollision::asOsgMatrix(centerOfObjectTransformationInWorldCoordinates)*parentInverse;
        // Call the callback, if registered.
        if (_mscl.size() > 0)
        {
            // Call only if position changed.
            const btVector3 delta(centerOfObjectTransformationInWorldCoordinates.getOrigin() - _transform.getOrigin());
            const btScalar eps((btScalar)(1e-5));
            const bool quiescent(osg::equivalent(delta[0], btScalar(0.), eps) &&
                osg::equivalent(delta[1], btScalar(0.), eps) &&
                osg::equivalent(delta[2], btScalar(0.), eps));
            if (!quiescent)
            {
                osgbDynamics::MotionStateCallbackList::iterator it;
                for (it = _mscl.begin(); it != _mscl.end(); ++it)
                    (**it)(centerOfObjectTransformationInWorldCoordinates);
            }
        }

        // _transform is the model-to-world transformation used to place collision shapes
        // in the physics simulation. Bullet queries this with getWorldTransform().
        _transform = centerOfObjectTransformationInWorldCoordinates;

        if (_tb == NULL)
            setWorldTransformInternal(osgbCollision::asBtTransform(localMat));
        else
        {
            char* addr(_tb->writeAddress());
            if (addr == NULL)
            {
                osg::notify(osg::WARN) << "VrmlMotionState: No TripleBuffer write address." << std::endl;
                return;
            }
            btScalar* fAddr = reinterpret_cast<btScalar*>(addr + _tbIndex);
            osgbCollision::asBtTransform(localMat).getOpenGLMatrix(fAddr);
        }
    }
    void
        VrmlMotionState::getWorldTransform(btTransform& worldTrans) const
    {
        worldTrans = _transform;
    }

    void
        VrmlMotionState::setWorldTransformInternal(const btTransform& worldTrans)
    {
        // Compute the transformation of the OSG visual representation.
        const osg::Matrix dt = osgbCollision::asOsgMatrix(worldTrans);
        const osg::Matrix col2ol = computeCOLocalToOsgLocal();
        const osg::Matrix t = col2ol * dt;

        if (_mt.valid())
            _mt->setMatrix(t);
        else if (_amt.valid())
            _amt->setMatrix(t);
    }

    osg::Matrix VrmlMotionState::computeCOLocalToOsgLocal() const
    {
        // Accound for center of mass and scale.
        const osg::Vec3 cs(_com[0] * _scale[0], _com[1] * _scale[1], _com[2] * _scale[2]);
        const osg::Matrix csMat = osg::Matrix::translate(-cs);

        const osg::Matrix scale = osg::Matrix::scale(_scale);

        // Return the concatenation of these.
        return(scale * csMat);
    }
    osg::Matrix VrmlMotionState::computeOsgLocalToCOLocal() const
    {
        // Accound for center of mass and scale.
        const osg::Vec3 cs(_com[0] * _scale[0], _com[1] * _scale[1], _com[2] * _scale[2]);
        const osg::Matrix csMat = osg::Matrix::translate(-cs);

        return(csMat);
    }
    osg::Matrix VrmlMotionState::computeOsgWorldToCOLocal() const
    {
        // Convert to OSG local coords...
        osg::Matrix l2w;
        if (_mt.valid())
            l2w = _mt->getMatrix();
        else if (_amt.valid())
            l2w = _amt->getMatrix();
        osg::Matrix w2l;
        w2l.invert(l2w);

        // ...and accound for center of mass and scale.
        osg::Matrix ol2col = computeOsgLocalToCOLocal();

        osg::Matrix scale = osg::Matrix::scale(_scale);

        // Return the concatenation of these.
        return(w2l * ol2col * scale);
    }
    osg::Matrix VrmlMotionState::computeOsgWorldToBulletWorld() const
    {
        // Compute OSG world coords to collision object local coords matrix...
        osg::Matrix ow2col = computeOsgWorldToCOLocal();

        // ...and convert out to Bullet world coords.
        btTransform bulletl2w;
        getWorldTransform(bulletl2w);
        osg::Matrix bl2w = osgbCollision::asOsgMatrix(bulletl2w);

        // Return the concatenation of these.
        return(ow2col * bl2w);
    }


    void
        VrmlMotionState::setTransform(osg::Transform* transform)
    {
        if (transform->asMatrixTransform() != NULL)
            _mt = transform->asMatrixTransform();
        else if (dynamic_cast<osgwTools::AbsoluteModelTransform*>(transform))
            _amt = static_cast<osgwTools::AbsoluteModelTransform*>(transform);
        else
            osg::notify(osg::WARN) << "VrmlMotionState: Unsupported transform type: " << transform->className() << std::endl;
    }

    osg::Transform*
        VrmlMotionState::getTransform()
    {
        if (_mt.valid())
            return(_mt.get());
        else if (_amt.valid())
            return(_amt.get());
        else
            return NULL;
    }

    const osg::Transform*
        VrmlMotionState::getTransform() const
    {
        if (_mt.valid())
            return(_mt.get());
        else if (_amt.valid())
            return(_amt.get());
        else
            return NULL;
    }

    void VrmlMotionState::setLocalTransform(const osg::Matrix localTrans) {
        _localMatrix = localTrans;
    }

    void
        VrmlMotionState::setParentTransform(const osg::Matrix parentTransformExcludingLocalCoordinates)
    {
        //    osg::notify( osg::ALWAYS) << "setParent" << m << std::endl;

        _parentTransform = osg::Matrix::orthoNormal(parentTransformExcludingLocalCoordinates);

        resetTransform();
    }

    osg::Matrix
        VrmlMotionState::getParentTransform() const
    {
        return(_parentTransform);
    }


    void
        VrmlMotionState::setCenterOfMass(const osg::Vec3& com)
    {
        _com = com;
        resetTransform();
    }
    osg::Vec3
        VrmlMotionState::getCenterOfMass() const
    {
        return(_com);
    }

    void
        VrmlMotionState::setScale(const osg::Vec3& scale)
    {
        _scale = scale;
        resetTransform();
    }
    osg::Vec3
        VrmlMotionState::getScale() const
    {
        return(_scale);
    }

    osgbDynamics::MotionStateCallbackList&
        VrmlMotionState::getCallbackList()
    {
        return(_mscl);
    }



    void
        VrmlMotionState::resetTransform()    {
        // Divide the center of mass by the scale, concatenate it with the parent transform,
        // then call setWorldTransform.
        // This creates the initial model-to-world transform for both the collision shape /
        // rigid body and the OSG visual representation.
        setWorldTransform(_transform);
    }


    void
        VrmlMotionState::registerTripleBuffer(osgbDynamics::TripleBuffer* tb)
    {
        _tb = tb;
        _tbIndex = tb->reserve(sizeof(btScalar) * 16);
    }

    void
        VrmlMotionState::updateTripleBuffer(const char* addr)
    {
        const btScalar* fAddr = reinterpret_cast<const btScalar*>(addr + _tbIndex);
        btTransform trans;
        trans.setFromOpenGLMatrix(fAddr);
        setWorldTransformInternal(trans);
    }

    bool TripleBufferMotionStateUpdate(osgbDynamics::MotionStateList& msl, osgbDynamics::TripleBuffer* tb)
    {
        const char* addr = tb->beginRead();
        if (addr == NULL)
            // No updated buffer is available. No valid data.
            return(false);

        osgbDynamics::MotionStateList::const_iterator it;
        for (it = msl.begin(); it != msl.end(); ++it)
            (*it)->updateTripleBuffer(addr);

        tb->endRead();
        return(true);
    }


