/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System.Collections.Generic;
using System.Windows;
using Autodesk.Revit.DB;
using System.Windows.Media.Media3D;
using utils;

namespace OpenFOAMInterface.BIM.OpenFOAM
{
    /// <summary>
    /// The class SnappyHexMeshDict heritates from abstract class FoamDict and contains all default attributes for this openFOAM-File.
    /// </summary>
    public class SnappyHexMeshDict : FOAMDict
    {
        ///Constant strings
        private const string nameGeometry = "name";
        private const string level = "level";
        private const string regions = "regions";

        /// <summary>
        /// Name of the STL
        /// </summary>
        private readonly string m_STLName;

        /// <summary>
        /// Point in 3d-Space that is used to seperate between outer and inner mesh in the snappyHexMesh-algorithmn
        /// </summary>
        private Vector3D m_LocationInMesh;

        /// <summary>
        /// Contains inlet and outlet as Faces
        /// </summary>
        private readonly Dictionary<KeyValuePair<string, Document>, KeyValuePair<List<Face>/*Face*/, Transform>> m_Faces;

        /// <summary>
        /// Simulation default => castellatedMeshControls.
        /// </summary>
        private Dictionary<string, object> m_SettingsCMC;

        //Default-Dictionaries in SnappyHexMeshDict
        private Dictionary<string, object> m_Geometry;
        private Dictionary<string, object> m_CastellatedMeshControls;
        private Dictionary<string, object> m_RefinementSurfaces;
        private Dictionary<string, object> m_RefinementRegions;

        //Geometry-Dictionary
        Dictionary<string, object> m_Regions;
        Dictionary<string, object> m_Stl;
        Dictionary<string, object> m_BoxGeometry;

        //Castellated-Dictionary
        Dictionary<string, object> m_StlRefinement;
        Dictionary<string, object> m_RegionsRefinementCastellated;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="version">Version-object.</param>
        /// <param name="path">Path to this file.</param>
        /// <param name="attributes">Additional attributes.</param>
        /// <param name="format">Ascii or Binary</param>
        /// <param name="settings">Settings-object</param>
        /// <param name="stlName">Name of the STL</param>
        /// <param name="faces">Outlet & Inlet as Faces in a Dictionary with name as Key.</param>
        public SnappyHexMeshDict(Version version, string path, Dictionary<string, object> attributes, SaveFormat format, string stlName, string stlWallName,
            Dictionary<KeyValuePair<string, Document>, KeyValuePair<List<Face>/*Face*/, Transform>> faces)
            : base("snappyHexMeshDict", "dictionary", version, path, attributes, format)
        {
            m_STLName = stlName;
            m_Faces = faces;

            m_Geometry = new Dictionary<string, object>();
            m_CastellatedMeshControls = new Dictionary<string, object>();
            m_RefinementSurfaces = new Dictionary<string, object>();
            m_RefinementRegions = new Dictionary<string, object>();


            m_Regions = new Dictionary<string, object>();
            m_Stl = new Dictionary<string, object>();
            m_BoxGeometry = new Dictionary<string, object>();

            m_StlRefinement = new Dictionary<string, object>();
            m_RegionsRefinementCastellated = new Dictionary<string, object>();

            m_SettingsCMC = m_DictFile["castellatedMeshControls"] as Dictionary<string, object>;

            InitAttributes();
        }

        /// <summary>
        /// Initialize attributes of this file.
        /// </summary>
        public override void InitAttributes()
        {
            InitGeometry();
            InitCastellatedMeshControls();

            FoamFile.Attributes.Add("castellatedMesh", m_DictFile["castellatedMesh"]);
            FoamFile.Attributes.Add("snap", m_DictFile["snap"]);
            FoamFile.Attributes.Add("addLayers", m_DictFile["addLayers"]);
            FoamFile.Attributes.Add("geometry", m_Geometry);
            FoamFile.Attributes.Add("castellatedMeshControls", m_CastellatedMeshControls);
            FoamFile.Attributes.Add("snapControls", m_DictFile["snapControls"]);
            FoamFile.Attributes.Add("addLayersControls", m_DictFile["addLayersControls"]);
            FoamFile.Attributes.Add("meshQualityControls", m_DictFile["meshQualityControls"]);
            FoamFile.Attributes.Add("debug", Exporter.Instance.settings.Debug);
            FoamFile.Attributes.Add("mergeTolerance", Exporter.Instance.settings.MergeTolerance);
        }

        /// <summary>
        /// Initialize the Geometry-Dictionary.
        /// </summary>
        private void InitGeometry()
        {
            InitGeometryRegions();

            bool singleSTLFile = false;
            if (!singleSTLFile)
            {

                foreach (Dictionary<string, object> region in m_Regions.Values)
                {
                    Dictionary<string, object> regionDict = new Dictionary<string, object>();
                    regionDict.Add("type", "triSurfaceMesh");
                    string regName = region["name"] as string;
                    Dictionary<string, object> nameDict = new Dictionary<string, object>();
                    nameDict.Add(regions, region);
                    regionDict.Add(nameGeometry, regName);
                    regionDict.Add(regName, nameDict);
                    string nameWithExtension = regName + ".stl";
                    m_Geometry.Add(nameWithExtension, regionDict);
                }
            }
            else
            {

                m_Stl.Add("type", "triSurfaceMesh");
                m_Stl.Add(nameGeometry, m_STLName);
                m_Stl.Add(regions, m_Regions);
                string nameWithExtension = m_STLName + ".stl";
                m_Geometry.Add(nameWithExtension, m_Stl);
            }
            Settings s = Exporter.Instance.settings;
            if (s.RefinementBoxOrigin[0] != 0)
            {

                Vector3D origin = new Vector3D(s.RefinementBoxOrigin.X, s.RefinementBoxOrigin.Y, s.RefinementBoxOrigin.Z);
                Vector3D e1 = new Vector3D(s.RefinementBoxX.X, s.RefinementBoxX.Y, s.RefinementBoxX.Z);
                Vector3D e2 = new Vector3D(s.RefinementBoxY.X, s.RefinementBoxY.Y, s.RefinementBoxY.Z);
                Vector3D e3 = new Vector3D(s.RefinementBoxZ.X, s.RefinementBoxZ.Y, s.RefinementBoxZ.Z);

                Vector3D vecs = new Vector3D(e1.Length, e2.Length, e3.Length);
                e1.Normalize();
                e2.Normalize();
                e3.Normalize();



                m_BoxGeometry.Add("type", "searchableRotatedBox");
                m_BoxGeometry.Add("span", vecs);
                m_BoxGeometry.Add("origin", origin);
                m_BoxGeometry.Add("e1", e1);
                m_BoxGeometry.Add("e2", e2);
                m_BoxGeometry.Add("e3", e3);

                m_Geometry.Add("boxRotated", m_BoxGeometry);
            }
        }

        /// <summary>
        /// Initialize the regions in the Geometry-Dictionary
        /// </summary>
        private void InitGeometryRegions()
        {
            string name;
            string wallName = "wallSTL";
            m_Regions.Add(wallName, new Dictionary<string, object> { { nameGeometry, wallName } });
            foreach (var face in m_Faces)
            {
                //face.Key.Key = Name + ID
                name = face.Key.Key;
                name = name.Replace(" ", "_");
                //m_Regions.Add(name, new Dictionary<string, object> { { nameGeometry, name } });
                m_Regions.Add(name, new Dictionary<string, object> { { nameGeometry, name } });
            }
            foreach (var entry in Exporter.Instance.settings.MeshResolution)
            {
                name = AutodeskHelperFunctions.GenerateNameFromElement(entry.Key);
                m_Regions.Add(name, new Dictionary<string, object> { { nameGeometry, name } });
            }
            foreach (var entry in Exporter.Instance.settings.m_InletElements)
            {
                name = AutodeskHelperFunctions.GenerateNameFromElement(entry);
                m_Regions.Add("Inlet_" + name, new Dictionary<string, object> { { nameGeometry, "Inlet_" + name } });
            }
            foreach (var entry in Exporter.Instance.settings.m_OutletElements)
            {
                name = AutodeskHelperFunctions.GenerateNameFromElement(entry);
                m_Regions.Add("Outlet_" + name, new Dictionary<string, object> { { nameGeometry, "Outlet_" + name } });
            }
        }

        /// <summary>
        /// Initialize the CastellatedMeshControl-Dictionary.
        /// </summary>
        private void InitCastellatedMeshControls()
        {
            InitRefinementSurfaces();
            InitLocationInMesh();
            List<string> addAttributes = new List<string> { "maxLocalCells", "maxGlobalCells", "minRefinementCells", "maxLoadUnbalance", "nCellsBetweenLevels", "features" };
            foreach (var s in addAttributes)
            {
                m_CastellatedMeshControls.Add(s, m_SettingsCMC[s]);
            }
            m_CastellatedMeshControls.Add("refinementSurfaces", m_RefinementSurfaces);
            m_CastellatedMeshControls.Add("resolveFeatureAngle", m_SettingsCMC["resolveFeatureAngle"]);
            m_CastellatedMeshControls.Add("refinementRegions", m_RefinementRegions);
            Vector3D tmp = new Vector3D(0, 0, 0);
            tmp.X = UnitUtils.ConvertFromInternalUnits(m_LocationInMesh.X, UnitTypeId.Meters);
            tmp.Y = UnitUtils.ConvertFromInternalUnits(m_LocationInMesh.Y, UnitTypeId.Meters);
            tmp.Z = UnitUtils.ConvertFromInternalUnits(m_LocationInMesh.Z, UnitTypeId.Meters);
            m_CastellatedMeshControls.Add("locationInMesh", tmp);
            m_CastellatedMeshControls.Add("allowFreeStandingZoneFaces", m_SettingsCMC["allowFreeStandingZoneFaces"]);
        }

        /// <summary>
        /// Initialize RefinementSurfaces in CastellatedMesh-Dictionary.
        /// </summary>
        private void InitRefinementSurfaces()
        {
            m_StlRefinement.Add(level, m_SettingsCMC["wallLevel"]);
            bool singleSTLFile = false;
            if (!singleSTLFile)
            {

                //m_StlRefinement.Add(regions, m_RegionsRefinementCastellated);
                //m_RefinementSurfaces.Add(m_STLName, m_RegionsRefinementCastellated);
                m_RefinementSurfaces.Add(m_STLName, m_StlRefinement);
                InitRegionsRefinement();
            }
            else
            {
                InitRegionsRefinement();
                m_StlRefinement.Add(regions, m_RegionsRefinementCastellated);
                m_RefinementSurfaces.Add(m_STLName, m_StlRefinement);
            }
        }

        /// <summary>
        /// Initialize Regions in RefinementSurfaces.
        /// </summary>
        private void InitRegionsRefinement()
        {
            Vector vec = new Vector();
            string name;

            //patchtype dict
            Dictionary<string, object> patchType = new Dictionary<string, object> { { "type", "patch" } };
            // Settings s = Exporter.Instance.settings;
            var s = Exporter.Instance.settings;
            if (s.RefinementBoxOrigin[0] != 0)
            {
                int lev = s.RefinementBoxLevel;
                m_RefinementRegions.Add("boxRotated", new Dictionary<string, object>() { { "mode", "inside" }, { "levels", "((0.0001 " + lev + "))" } });
            }

            foreach (var face in m_Faces)
            {
                name = face.Key.Key;
                name = name.Replace(" ", "_");
                if (name.Contains("Inlet") || name.Contains("Zuluft"))
                {
                    vec = (Vector)m_SettingsCMC["inletLevel"];
                }
                else if (name.Contains("Outlet") || name.Contains("Abluft"))
                {
                    vec = (Vector)m_SettingsCMC["outletLevel"];
                }
                m_RefinementSurfaces.Add(name, new Dictionary<string, object>() { { level, vec }, { "patchInfo", patchType } });
                //m_RegionsRefinementCastellated.Add(name, new Dictionary<string, object>() { { level, vec} });
            }
            foreach (var entry in Exporter.Instance.settings.MeshResolution)
            {
                name = AutodeskHelperFunctions.GenerateNameFromElement(entry.Key);
                //if(name.Contains("Zuluft") || name.Contains("Abluft") || name.Contains("Outlet") || name.Contains("Inlet"))
                //{
                //    //name = "Terminal_" + name;
                //    m_RegionsRefinementCastellated[name] = ;
                //    continue;
                //}
                vec = new Vector(entry.Value, entry.Value);
                //m_RefinementSurfaces.Add(name, new Dictionary<string, object>() { { level, vec }, { "patchInfo", "wall" } });
                m_RefinementSurfaces.Add(name, new Dictionary<string, object>() { { level, vec } });
            }
            foreach (var entry in Exporter.Instance.settings.m_InletElements)
            {
                name = AutodeskHelperFunctions.GenerateNameFromElement(entry);
                vec = (Vector)m_SettingsCMC["inletLevel"];
                m_RefinementSurfaces.Add("Inlet_" + name, new Dictionary<string, object>() { { level, vec }, { "patchInfo", patchType } });
            }
            foreach (var entry in Exporter.Instance.settings.m_OutletElements)
            {
                name = AutodeskHelperFunctions.GenerateNameFromElement(entry);
                vec = (Vector)m_SettingsCMC["outletLevel"];
                m_RefinementSurfaces.Add("Outlet_" + name, new Dictionary<string, object>() { { level, vec }, { "patchInfo", patchType } });
            }
        }

        /// <summary>
        /// Initialize Vector.
        /// </summary>
        private void InitLocationInMesh()
        {
            m_LocationInMesh = Exporter.Instance.settings.LocationInMesh;
        }
    }
}