#pragma once
#include <lib/core/interfaces/IInfoboard.h>
#include <osg/Node>
#include <osg/ref_ptr>

typedef core::interface::IInfoboard<std::string, osg::ref_ptr<osg::Node>> OsgInfoboard;
