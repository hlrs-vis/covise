#pragma once
#include <osg/MatrixTransform>

// ---------------------------------------------------------------------------
// local-to-root matrix WITHOUT osg::computeLocalToWorld: the objects root is
// a ClipNode (an osg::Camera), and the free osg::computeLocalToWorld skips
// absolute/root cameras -- once the root has a parent (i.e. after scene setup)
// it silently drops the whole chain and returns identity.  Instead walk the
// path manually and apply each node's virtual computeLocalToWorldMatrix, which
// is correct for both MatrixTransform and Camera and stable across frames.
// The path runs root->...->node; the root itself maps root space to itself.
// ---------------------------------------------------------------------------
osg::Matrix localToWorldBelowRoot(const osg::NodePath &path);
