/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2009 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

#include <vpb/SpatialProperties>
//#include <vpb/BuildLog>

#include <osg/Notify>
//#include <osg/io_utils>

#include <cpl_string.h>
#include <ogr_spatialref.h>

using namespace vpb;

CoordinateSystemType vpb::getCoordinateSystemType(const osg::CoordinateSystemNode *lhs)
{
    if (!lhs)
        return PROJECTED;

    // set up LHS SpatialReference
    char *projection_string = strdup(lhs->getCoordinateSystem().c_str());
    char *importString = projection_string;

    OGRSpatialReference lhsSR;
    lhsSR.importFromWkt(&importString);

    //  log(osg::INFO,"getCoordinateSystemType(%s)",projection_string);
    //  log(osg::INFO,"    lhsSR.IsGeographic()=%d",lhsSR.IsGeographic());
    //  log(osg::INFO,"    lhsSR.IsProjected()=%d",lhsSR.IsProjected());
    //  log(osg::INFO,"    lhsSR.IsLocal()=%d",lhsSR.IsLocal());

    free(projection_string);

    //if (strcmp(lhsSR.GetRoot()->GetValue(),"GEOCCS")==0) log(osg::INFO,"    lhsSR. is GEOCENTRIC ");

    if (strcmp(lhsSR.GetRoot()->GetValue(), "GEOCCS") == 0)
        return GEOCENTRIC;
    if (lhsSR.IsGeographic())
        return GEOGRAPHIC;
    if (lhsSR.IsProjected())
        return PROJECTED;
    if (lhsSR.IsLocal())
        return LOCAL;
    return PROJECTED;
}

std::string vpb::coordinateSystemStringToWTK(const std::string &coordinateSystem)
{
    std::string wtkString;

    CPLErrorReset();

    OGRSpatialReferenceH hSRS = OSRNewSpatialReference(NULL);
    if (OSRSetFromUserInput(hSRS, coordinateSystem.c_str()) == OGRERR_NONE)
    {
        char *pszResult = NULL;
        OSRExportToWkt(hSRS, &pszResult);

        if (pszResult)
            wtkString = pszResult;

        CPLFree(pszResult);
    }
    else
    {
        //  log(osg::WARN,"Warning: coordinateSystem string not recognised.");
    }

    OSRDestroySpatialReference(hSRS);

    return wtkString;
}

double vpb::getLinearUnits(const osg::CoordinateSystemNode *lhs)
{
    // set up LHS SpatialReference
    char *projection_string = strdup(lhs->getCoordinateSystem().c_str());
    char *importString = projection_string;

    OGRSpatialReference lhsSR;
    lhsSR.importFromWkt(&importString);

    free(projection_string);

    char *str;
    double result = lhsSR.GetLinearUnits(&str);
    //  log(osg::INFO,"lhsSR.GetLinearUnits(%s) %f",str,result);

    //  log(osg::INFO,"lhsSR.IsGeographic() %d",lhsSR.IsGeographic());
    //  log(osg::INFO,"lhsSR.IsProjected() %d",lhsSR.IsProjected());
    //  log(osg::INFO,"lhsSR.IsLocal() %d",lhsSR.IsLocal());

    return result;
}

bool vpb::areCoordinateSystemEquivalent(const osg::CoordinateSystemNode *lhs, const osg::CoordinateSystemNode *rhs)
{
    // if ptr's equal the return true
    if (lhs == rhs)
        return true;

    // if one CS is NULL then true false
    if (!lhs || !rhs)
    {
        //  log(osg::INFO,"areCoordinateSystemEquivalent lhs=%s  rhs=%s return true",lhs,rhs);
        return false;
    }

    //log(osg::INFO,"areCoordinateSystemEquivalent lhs=%s rhs=%s",lhs->getCoordinateSystem().c_str(),rhs->getCoordinateSystem().c_str());

    // use compare on ProjectionRef strings.
    if (lhs->getCoordinateSystem() == rhs->getCoordinateSystem())
        return true;

    // set up LHS SpatialReference
    char *projection_string = strdup(lhs->getCoordinateSystem().c_str());
    char *importString = projection_string;

    OGRSpatialReference lhsSR;
    lhsSR.importFromWkt(&importString);

    free(projection_string);

    // set up RHS SpatialReference
    projection_string = strdup(rhs->getCoordinateSystem().c_str());
    importString = projection_string;

    OGRSpatialReference rhsSR;
    rhsSR.importFromWkt(&importString);

    free(projection_string);

    int result = lhsSR.IsSame(&rhsSR);

#if 0
    int result2 = lhsSR.IsSameGeogCS(&rhsSR);

     log(osg::NOTICE)<<"areCoordinateSystemEquivalent "<<std::endl
              <<"LHS = "<<lhs->getCoordinateSystem()<<std::endl
              <<"RHS = "<<rhs->getCoordinateSystem()<<std::endl
              <<"result = "<<result<<"  result2 = "<<result2);
#endif
    return result ? true : false;
}

void SpatialProperties::computeExtents()
{
    _extents.init();
    _extents.expandBy(osg::Vec3(0.0, 0.0, 0.0) * _geoTransform);

    // get correct extent if a vector format is used
    if (_dataType == VECTOR)
        _extents.expandBy(osg::Vec3(_numValuesX - 1, _numValuesY - 1, 0.0) * _geoTransform);
    else
        _extents.expandBy(osg::Vec3(_numValuesX, _numValuesY, 0.0) * _geoTransform);

    _extents._isGeographic = getCoordinateSystemType(_cs.get()) == GEOGRAPHIC;

    // log(osg::INFO,"DataSet::SpatialProperties::computeExtents() is geographic %d",_extents._isGeographic);
}

bool SpatialProperties::equivalentCoordinateSystem(const SpatialProperties &sp) const
{
    return areCoordinateSystemEquivalent(_cs.get(), sp._cs.get());
}

bool SpatialProperties::intersects(const SpatialProperties &sp) const
{
#if 0
    osg::notify(osg::NOTICE)<<"    SpatialProperties::intersects(sp) : _extents.intersects(sp._extents)="<<_extents.intersects(sp._extents)<<std::endl;
    osg::notify(osg::NOTICE)<<"                                        _extents.valid()="<<_extents.valid()<<std::endl;
    osg::notify(osg::NOTICE)<<"                                        sp._extents.valid()="<<sp._extents.valid()<<std::endl;
    osg::notify(osg::NOTICE)<<"                                        _extents.min()="<<_extents._min<<" max()"<<_extents._max<<std::endl;
    osg::notify(osg::NOTICE)<<"                                        sp._extents.min()="<<sp._extents._min<<" max()"<<sp._extents._max<<std::endl;
#endif
    return _extents.intersects(sp._extents);
}

bool SpatialProperties::compatible(const SpatialProperties &sp) const
{
#if 0
    osg::notify(osg::NOTICE)<<"  SpatialProperties::compatible(sp) : equivalentCoordinateSystem(sp)="<<equivalentCoordinateSystem(sp)<<std::endl;
    osg::notify(osg::NOTICE)<<"                                    : intersects(sp)="<<intersects(sp)<<std::endl;
#endif
    return equivalentCoordinateSystem(sp) && intersects(sp);
}

double SpatialProperties::computeResolutionRatio(const SpatialProperties &sp) const
{
    return sp.computeResolution() / computeResolution();
}

double SpatialProperties::computeResolution() const
{
    double resolutionX = (_extents.xMax() - _extents.xMin()) / double(_numValuesX - 1);
    double resolutionY = (_extents.yMax() - _extents.yMin()) / double(_numValuesY - 1);
    return sqrt(resolutionX * resolutionX + resolutionY * resolutionY);
}
