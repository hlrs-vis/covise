/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

void generateTube(float *verts,
                  int vertNb,
                  float *veloGrad,
                  double radius,
                  UniGeom *ugeom)
{
#if 0
  for (int v=0; v<vertNb-1; v++) {
    ..
  }
#else
    float vertices[1000] = {
        -1, 0, 0,
        0, 0, 1,
        1, 0, 0,
        0, 0, -1,
        -1, 1, 0,
        0, 1, 1,
        1, 1, 0,
        0, 1, -1
    };
    ugeom->addVertices(vertices, 8);
    //ugeom->addVertices(vertices, 4);
    int face0[4] = { 0, 1, 5, 4 };
    int face1[4] = { 1, 2, 6, 5 };
    int face2[4] = { 2, 3, 7, 6 };
    int face3[4] = { 3, 0, 4, 7 };
    ugeom->addPolygon(4, face0);
// ugeom->addPolygon(4, face1);
//ugeom->addPolygon(4, face2);
//ugeom->addPolygon(4, face3);

#if 0
  unsigned char imageBuf[5*2*4];
  DO_PixelImage *img = new DO_PixelImage("tmp", 5, 2, 0, 0, (const char *) imageBuf);

  Texture();

  delete img;
#endif

#endif
}

void vorticity_transport_impl(UniSys *us, Unstructured *unst, int compV,
                              UniGeom *ugeomCoreLines,
                              float startTime,
                              float integrationTime,
                              int integStepsMax,
                              UniGeom *ugTubes,
                              UniGeom *ugLines)
{
    // ### TODO: param
    bool forward = false;
    bool unsteady = false; // ###
    char velocity_file[256] = "";
    int smoothingRange = 1;
    double radius = 0.01;

    ugTubes->createObj(UniGeom::GT_POLYHEDRON);
    ugLines->createObj(UniGeom::GT_LINE);

    if (unsteady)
    {
        // set unst to transient mode
        unst->setTransientFile(velocity_file,
                               true // verbose
                               );
    }

    float *verts = new float[(integStepsMax + 1) * 3];
    float *veloGrad = new float[(integStepsMax + 1) * 3 * 3];

    for (int v = 0; v < ugeomCoreLines->getVertexCnt(); v++)
    {

        if (!(v % 100))
        {
            char buf[256];
            sprintf(buf, "pathline for vertex %d", v);
            us->moduleStatus(buf, (int)((100.0 * v) / ugeomCoreLines->getVertexCnt()));
        }

        vec3 start;
        ugeomCoreLines->getVertex(v, start);

        // compute path line
        double itime = integrationTime / integStepsMax;
        double timeTot = 0.0;
        double ilength = 1e20;
        int stepNbEff = 0;
        verts[(0) * 3 + 0] = start[0];
        verts[(0) * 3 + 1] = start[1];
        verts[(0) * 3 + 2] = start[2];

        unst->gradient(compV, start, &veloGrad[0 * 9], startTime, smoothingRange);

        for (int step = 0; step < integStepsMax; step++)
        {

            double ds;
            double dt = unst->integrate(start, forward, itime, // 1e9,
                                        ilength,
                                        1000000, // upper limit of steps
                                        4, false,
                                        (forward ? startTime + timeTot : startTime - timeTot),
                                        &ds);
            timeTot += dt;

            verts[(step + 1) * 3 + 0] = start[0];
            verts[(step + 1) * 3 + 1] = start[1];
            verts[(step + 1) * 3 + 2] = start[2];
            unst->gradient(compV, start, &veloGrad[(step + 1) * 9], (forward ? startTime + timeTot : startTime - timeTot), smoothingRange);
            stepNbEff++;

            if (dt == 0.0)
            {
                break;
            }
        }

        ugLines->addPolyline(verts, NULL, stepNbEff + 1);

        // generate tube
        generateTube(verts, stepNbEff + 1, veloGrad, radius, ugTubes);
        delete[] verts;
        delete[] veloGrad;

        //###printf("%d\n", v);
    }

    if (unsteady)
    {
        printf("total count of mmap() calls: %d (should be as low as possible)\n", unst->getTransientFileMapNb());
        unst->unsetTransientFile();
    }

    ugTubes->assignObj("striped path lines");
    ugLines->assignObj("path lines");
}
