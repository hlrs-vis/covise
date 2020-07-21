///////////////////////////////////////////////////////////////////////////////////////////////////
// LibGizmo
// File Name : 
// Creation : 10/01/2012
// Author : Cedric Guillemet
// Description : LibGizmo
//
///Copyright (C) 2012 Cedric Guillemet
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
//of the Software, and to permit persons to whom the Software is furnished to do
///so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
///FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// 


#ifndef ZCOLLISIONSUTILS_H__
#define ZCOLLISIONSUTILS_H__


///////////////////////////////////////////////////////////////////////////////////////////////////


inline bool CollisionClosestPointOnSegment( const tvector3 & point, const tvector3 & vertPos1, const tvector3 & vertPos2, tvector3& res )
{

    tvector3 c = point - vertPos1;
    tvector3 V;

   V.Normalize(vertPos2 - vertPos1);
    float d = (vertPos2 - vertPos1).Length();
    float t = V.Dot(c);

    if (t < 0)
   {
      return false;//vertPos1;
   }

    if (t > d)
   {
      return false;//vertPos2;
   }

    res = vertPos1 + V * t;
    return true;
}

inline tvector3 CollisionClosestPointOnSegment( const tvector3 & point, const tvector3 & vertPos1, const tvector3 & vertPos2 )
{

    tvector3 c = point - vertPos1;
    tvector3 V;

   V.Normalize(vertPos2 - vertPos1);
    float d = (vertPos2 - vertPos1).Length();
    float t = V.Dot(c);

    if (t < 0)
   {
      return vertPos1;
   }

    if (t > d)
   {
      return vertPos2;
   }

    return vertPos1 + V * t;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 CollisionClosestPointOnTriangle( tvector3 & point, tvector3 & vertPos1, tvector3 & vertPos2, tvector3 & vertPos3 )
{
    tvector3  Rab = CollisionClosestPointOnSegment( point, vertPos1, vertPos2 );
    tvector3  Rbc = CollisionClosestPointOnSegment( point, vertPos2, vertPos3 );
    tvector3  Rca = CollisionClosestPointOnSegment( point, vertPos3, vertPos1 );

    float  dRab = (point - Rab).Length();
    float  dRbc = (point - Rbc).Length();
    float  dRca = (point - Rca).Length();

    if( (dRab <  dRbc) && ( dRab < dRca ))
    {
        return Rab;
    }
    else{
        if( (dRbc < dRab) && ( dRbc < dRca ))
        {
            return Rbc;
        }
        else{
            return Rca;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

//inline bool isRayAABBoxIntersect(const tvector3& aMinB, const tvector3& aMaxB, const tvector3& aOrigin, const tvector3& aDir)
#define NUMDIM	3

inline char HitBoundingBox(float *minB, float *maxB, float *origin, float *dir,float *coord)
{
	char inside = TRUE;
	char quadrant[NUMDIM];
	register int i;
	int whichPlane;
	float maxT[NUMDIM];
	float candidatePlane[NUMDIM];

	/* Find candidate planes; this loop can be avoided if
   	rays cast all from the eye(assume perpsective view) */
	for (i=0; i<NUMDIM; i++)
		if(origin[i] < minB[i]) {
			quadrant[i] = 1;
			candidatePlane[i] = minB[i];
			inside = FALSE;
		}else if (origin[i] > maxB[i]) {
			quadrant[i] = 0;
			candidatePlane[i] = maxB[i];
			inside = FALSE;
		}else	{
			quadrant[i] = 2;
		}

	/* Ray origin inside bounding box */
	if(inside)	{
		coord = origin;
		return (TRUE);
	}


	/* Calculate T distances to candidate planes */
	for (i = 0; i < NUMDIM; i++)
		if (quadrant[i] != 2 && dir[i] !=0.)
			maxT[i] = (candidatePlane[i]-origin[i]) / dir[i];
		else
			maxT[i] = -1.;

	/* Get largest of the maxT's for final choice of intersection */
	whichPlane = 0;
	for (i = 1; i < NUMDIM; i++)
		if (maxT[whichPlane] < maxT[i])
			whichPlane = i;

	/* Check final candidate actually inside box */
	if (maxT[whichPlane] < 0.) return (FALSE);
	for (i = 0; i < NUMDIM; i++)
		if (whichPlane != i) {
			coord[i] = origin[i] + maxT[whichPlane] *dir[i];
			if (coord[i] < minB[i] || coord[i] > maxB[i])
				return (FALSE);
		} else {
			coord[i] = candidatePlane[i];
		}
	return (TRUE);				/* ray hits box */
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool AABBOverlapsSphere ( float *bMin, float *bMax, const float r, float* C )
{

    float s, d = 0;

    //find the square of the distance
    //from the sphere to the box
    for( long i=0 ; i<3 ; i++ )
    {

        if( C[i] < bMin[i] )
        {

            s = C[i] - bMin[i];
            d += s*s;

        }

        else if( C[i] > bMax[i] )
        {

            s = C[i] - bMax[i];
            d += s*s;

        }

    }
    return d <= r*r;

}

inline bool SphereOverlapsSphere (const tvector4& sph1, const tvector4& sph2)
{
	float distSq = SquaredDistance( sph1, sph2 );
	float sumSquaredRadius = (sph1.w * sph1.w) + (sph2.w * sph2.w);

	return (distSq <= sumSquaredRadius);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool isRayAABBoxIntersect(const tvector3& aMinB, const tvector3& aMaxB, const tvector3& aOrigin, const tvector3& aDir)
{
    tvector3 dmin(aMinB,aOrigin);
    tvector3 dmax(aMaxB,aOrigin);

    int    inside=0;
    float    tmax=-1.f;

    if (dmin.x>0.0f && aDir.x>0.0f)
        tmax=dmin.x/aDir.x;
    else
        if (dmax.x<0.0f && aDir.x<0.0f)
            tmax=dmax.x/aDir.x;
        else
            inside++;

    if (dmax.y<0.0f && aDir.y<0.0f && tmax*aDir.y> dmax.y)
        tmax=dmax.y/aDir.y;
    else
        if (dmin.y>0.0f && aDir.y>0.0f && tmax*aDir.y<dmin.y)
            tmax=dmin.y/aDir.y;
        else
            inside++;

    if (dmin.z>0.0f && aDir.z>0.0f && tmax*aDir.z<dmin.z)
        tmax=dmin.z/aDir.z;
    else
        if (dmax.z<0.0f && aDir.z<0.0f && tmax*aDir.z>dmax.z)
            tmax= dmax.z/aDir.z;
        else
            inside++;

    if (inside==3)
        return TRUE;

    if (tmax<0.0f)
        return FALSE;

    // Check final candidate actually inside box
    tvector3 V = aDir * tmax;
    //Mul(V,aDir,tmax);
    dmin-=V;
    dmax-=V;
    dmin-=tvector3::Epsilon;
    dmax+=tvector3::Epsilon;

    if (dmin.x>0.0f || dmin.y>0.0f || dmin.z>0.0f ||
        dmax.x<0.0f || dmax.y <0.0f || dmax.z<0.0f)
        return FALSE;

    return TRUE;    // ray hits box
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float IntersectRayPlane(const tvector3 & rOrigin, const tvector3& rVector, const tvector3& pOrigin, const tvector3 & pNormal)
{

  float d = - pNormal.Dot(pOrigin);

  float numer = pNormal.Dot(rOrigin) + d;
  float denom = pNormal.Dot(rVector);

  if (denom == 0)  // normal is orthogonal to vector, cant intersect
  {
      return (-1.0f);
  }

  return -(numer / denom);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
// Name  : intersectRaySphere()
// Input : rO - origin of ray in world space
//         rV - vector describing direction of ray in world space
//         sO - Origin of sphere
//         sR - radius of sphere
// Notes : Normalized directional vectors expected
// Return: distance to sphere in world units, -1 if no intersection.
// -----------------------------------------------------------------------

inline float IntersectRaySphere(const tvector3 & rO, const tvector3 & rV, const tvector3 & sO, float sR)
{

   tvector3 Q = sO-rO;

   //float c = Q.Length();
   float v = Q.Dot(rV);
   float d = sR*sR - (Q.LengthSq() - v*v);

   // If there was no intersection, return -1
   if (d < 0.0)
    {
        return (-1.0f);
    }

   // Return the distance to the [first] intersecting point
   return (v - MathSqrt(d));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
// Name  : CheckPointInTriangle()
// Input : point - point we wish to check for inclusion
//         a - first vertex in triangle
//         b - second vertex in triangle
//         c - third vertex in triangle
// Notes : Triangle should be defined in clockwise order a,b,c
// Return: TRUE if point is in triangle, FALSE if not.
// -----------------------------------------------------------------------

#define ZTRIin(a) ((tulong&) a)

inline bool CheckPointInTriangle(const tvector3 & point, const tvector3 & pa, const tvector3 & pb, const tvector3& pc)
{
    // old Method

    tvector3 e10 = pb-pa;
    tvector3 e20 = pc-pa;



    float a = e10.Dot(e10);
    float b = e10.Dot(e20);
    float c = e20.Dot(e20);
    float ac_bb = (a*c)-(b*b);

    tvector3 vp(point.x-pa.x, point.y - pa.y, point.z - pa.z);


    float d = vp.Dot(e10);
    float e = vp.Dot(e20);
    float x = (d*c) - (e*b);
    float y = (e*a) - (d*b);
    float z = x+y-ac_bb;

    return (bool)((( ZTRIin(z) & ~(ZTRIin(x)|ZTRIin(y)) ) & 0x80000000)>=1);

}

///////////////////////////////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
// Name  : CheckPointInTriangle()
// Input : point - point we wish to check for inclusion
//         sO - Origin of sphere
//         sR - radius of sphere
// Notes :
// Return: TRUE if point is in sphere, FALSE if not.
// -----------------------------------------------------------------------

inline bool CheckPointInSphere(const tvector3 & point, const tvector3 & sO, float sR)
{

    float d = (point-sO).Length();

    if(d<= sR)
    {
        return true;
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#define EPSILON 0.000001
#define CROSS(dest,v1,v2) \
    dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
    dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
    dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
    dest[0]=v1[0]-v2[0]; \
    dest[1]=v1[1]-v2[1]; \
    dest[2]=v1[2]-v2[2];

inline int intersect_triangle(float orig[3], float dir[3],
                   float vert0[3], float vert1[3], float vert2[3],
                   float *t, float *u, float *v)
{
    float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
    float det,inv_det;

    /* find vectors for two edges sharing vert0 */
    SUB(edge1, vert1, vert0);
    SUB(edge2, vert2, vert0);

    /* begin calculating determinant - also used to calculate U parameter */
    CROSS(pvec, dir, edge2);

    /* if determinant is near zero, ray lies in plane of triangle */
    det = DOT(edge1, pvec);

#ifdef TEST_CULL           /* define TEST_CULL if culling is desired */
    if (det < EPSILON)
        return 0;

    /* calculate distance from vert0 to ray origin */
    SUB(tvec, orig, vert0);

    /* calculate U parameter and test bounds */
    *u = DOT(tvec, pvec);
    if (*u < 0.0 || *u > det)
        return 0;

    /* prepare to test V parameter */
    CROSS(qvec, tvec, edge1);

    /* calculate V parameter and test bounds */
    *v = DOT(dir, qvec);
    if (*v < 0.0 || *u + *v > det)
        return 0;

    /* calculate t, scale parameters, ray intersects triangle */
    *t = DOT(edge2, qvec);
    inv_det = 1.0 / det;
    *t *= inv_det;
    *u *= inv_det;
    *v *= inv_det;
#else                    /* the non-culling branch */
    if (det > -EPSILON && det < EPSILON)
        return 0;
    inv_det = 1.0f / det;

    /* calculate distance from vert0 to ray origin */
    SUB(tvec, orig, vert0);

    /* calculate U parameter and test bounds */
    *u = DOT(tvec, pvec) * inv_det;
    if (*u < 0.0 || *u > 1.0)
        return 0;

    /* prepare to test V parameter */
    CROSS(qvec, tvec, edge1);

    /* calculate V parameter and test bounds */
    *v = DOT(dir, qvec) * inv_det;
    if (*v < 0.0 || *u + *v > 1.0)
        return 0;

    /* calculate t, ray intersects triangle */
    *t = DOT(edge2, qvec) * inv_det;
#endif
    return 1;
}


inline bool IsPointInCone(const tvector3& point, const tvector3& conepos, const tvector3& conedir, float length, float radius)
{
	tvector3 topToPoint = point-conepos;

	// conedir is normalized

	float dt = topToPoint.Dot(conedir);

	if ((dt <0) || (dt>length))
		return false;

	float dist = Distance(point, conepos + (conedir*dt) );

	float ht = LERP(0, radius, dt/length);

	if (dist > ht)
		return false;

	return true;
}

#endif
