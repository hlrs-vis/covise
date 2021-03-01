/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System;
using System.Collections.Generic;

namespace BIM.OpenFOAMExport.OpenFOAM
{
    /// <summary>
    /// MeshDict is in use for meshing with cfMesh.
    /// </summary>
    public class MeshDict : FOAMDict
    {
        public MeshDict(Version version, string path, Dictionary<string, object> attributes, SaveFormat format)
            : base("meshDict", "dictionary", version, path, attributes, format)
        {

        }

        public override void InitAttributes()
        {
            throw new NotImplementedException();
        }
    }
}
