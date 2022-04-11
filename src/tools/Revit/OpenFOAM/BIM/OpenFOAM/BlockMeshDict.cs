/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System.Windows.Media.Media3D;
using System.Collections.Generic;
using System.Collections;
using System;

namespace OpenFOAMInterface.BIM.OpenFOAM
{
    /// <summary>
    /// The BlockMeshDict-Class contains all attributes for blockMesh in Openfoam.
    /// </summary>
    public class BlockMeshDict : FOAMDict
    {
        /// <summary>
        /// Cellsize for the boundingbox
        /// </summary>
        private Vector3D m_CellSize;

        /// <summary>
        /// Simplegrading vector
        /// </summary>
        private Vector3D m_SimpleGrading;

        /// <summary>
        /// Point in 3d-Space for boundingbox
        /// </summary>
        private Vector3D m_VecLowerEdgeLeft;

        /// <summary>
        /// Point in 3d-Space for boundingbox
        /// </summary>
        private Vector3D m_VecUpperEdgeRight;

        /// <summary>
        /// Vertices for Boundingbox
        /// </summary>
        private readonly ArrayList m_Vertices;

        /// <summary>
        /// Edges-Dict
        /// </summary>
        private readonly ArrayList m_Edges;

        /// <summary>
        /// MergePatchPair-Dict
        /// </summary>        
        private readonly ArrayList m_MergePatchPair;

        /// <summary>
        /// Blocks-Dict
        /// </summary>
        private readonly ArrayList m_Blocks;

        /// <summary>
        /// Boundary
        /// </summary>        
        private readonly ArrayList m_Boundary;

        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="version">Version-object.</param>
        /// <param name="path">Path to this File.</param>
        /// <param name="attributes">Additional attributes.</param>
        /// <param name="format">Ascii or Binary.</param>
        /// <param name="settings"></param>
        /// <param name="vecLowerEdgeLeft">3d-point.</param>
        /// <param name="vecUpperEdgeRight">3d-point.</param>
        public BlockMeshDict(Version version, string path, Dictionary<string, object> attributes, SaveFormat format, Vector3D vecLowerEdgeLeft, Vector3D vecUpperEdgeRight)
            : base("blockMeshDict", "dictionary", version, path, attributes, format)
        {

            m_VecLowerEdgeLeft = vecLowerEdgeLeft;
            m_VecUpperEdgeRight = vecUpperEdgeRight;


            m_Vertices = new ArrayList();
            m_Edges = new ArrayList();
            m_MergePatchPair = new ArrayList();
            m_Blocks = new ArrayList();
            m_Boundary = new ArrayList();

            InitAttributes();
        }

        /// <summary>
        /// Initialize attributes.
        /// </summary>
        public override void InitAttributes()
        {
            m_SimpleGrading = (Vector3D)m_DictFile["simpleGrading"];
            EnlargeBoundingboxVector(1);
            m_CellSize = (Vector3D)m_DictFile["cellSize"];

            InitBoundingboxFromPoints();
            if (m_CellSize.Length == 0)
            {
                InitDefaultCellSize();
            }
            InitBlocks();
            InitEdges();
            InitBoundary();
            InitMergePatchPair();

            FoamFile.Attributes.Add("vertices", m_Vertices);
            FoamFile.Attributes.Add("blocks", m_Blocks);
            FoamFile.Attributes.Add("edges", m_Edges);
            FoamFile.Attributes.Add("boundary", m_Boundary);
            FoamFile.Attributes.Add("mergePatchPair", m_MergePatchPair);
        }

        /// <summary>
        /// Initialize vertices with two points in 3D-Space.
        /// </summary>
        private void InitBoundingboxFromPoints()
        {
            Settings s = Exporter.Instance.settings;
            if (s.DomainX.IsZeroLength())
            {
                m_Vertices.Add(m_VecLowerEdgeLeft);
                m_Vertices.Add(new Vector3D(m_VecUpperEdgeRight.X, m_VecLowerEdgeLeft.Y, m_VecLowerEdgeLeft.Z));
                m_Vertices.Add(new Vector3D(m_VecUpperEdgeRight.X, m_VecUpperEdgeRight.Y, m_VecLowerEdgeLeft.Z));
                m_Vertices.Add(new Vector3D(m_VecLowerEdgeLeft.X, m_VecUpperEdgeRight.Y, m_VecLowerEdgeLeft.Z));
                m_Vertices.Add(new Vector3D(m_VecLowerEdgeLeft.X, m_VecLowerEdgeLeft.Y, m_VecUpperEdgeRight.Z));
                m_Vertices.Add(new Vector3D(m_VecUpperEdgeRight.X, m_VecLowerEdgeLeft.Y, m_VecUpperEdgeRight.Z));
                m_Vertices.Add(m_VecUpperEdgeRight);
                m_Vertices.Add(new Vector3D(m_VecLowerEdgeLeft.X, m_VecUpperEdgeRight.Y, m_VecUpperEdgeRight.Z));

            }
            else
            { 
                // m_Vertices.Add(new Vector3D(s.DomainOrigin.X, s.DomainOrigin.Y, s.DomainOrigin.Z));
                // Autodesk.Revit.DB.XYZ tmp = s.DomainOrigin+s.DomainX;
                // m_VecLowerEdgeLeft = new Vector3D(tmp.X, tmp.Y, tmp.Z);
                // m_Vertices.Add(new Vector3D(tmp.X,tmp.Y,tmp.Z));
                // tmp = s.DomainOrigin + s.DomainX + s.DomainY;
                // m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                // tmp = s.DomainOrigin + s.DomainY;
                // m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                // tmp = s.DomainOrigin + s.DomainZ;
                // m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                // tmp = s.DomainOrigin + s.DomainZ + s.DomainX;
                // m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                // tmp = s.DomainOrigin + s.DomainZ + s.DomainX + s.DomainY;
                // m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                // m_VecUpperEdgeRight = new Vector3D(tmp.X, tmp.Y, tmp.Z);
                // tmp = s.DomainOrigin + s.DomainZ + s.DomainY;
                // m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                Autodesk.Revit.DB.XYZ origin = s.DomainOrigin;
                Autodesk.Revit.DB.XYZ domainX = s.DomainX; 
                Autodesk.Revit.DB.XYZ domainY = s.DomainY; 
                Autodesk.Revit.DB.XYZ domainZ = s.DomainZ; 
                m_Vertices.Add(new Vector3D(origin.X, origin.Y, origin.Z));
                Autodesk.Revit.DB.XYZ tmp = origin + domainX;
                m_VecLowerEdgeLeft = new Vector3D(tmp.X, tmp.Y, tmp.Z);
                m_Vertices.Add(new Vector3D(tmp.X,tmp.Y,tmp.Z));
                tmp = origin + domainX + domainY;
                m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                tmp = origin + domainY;
                m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                tmp = origin + domainZ;
                m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                tmp = origin + domainZ + domainX;
                m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                tmp = origin + domainZ + domainX + domainY;
                m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
                m_VecUpperEdgeRight = new Vector3D(tmp.X, tmp.Y, tmp.Z);
                tmp = origin + domainZ + domainY;
                m_Vertices.Add(new Vector3D(tmp.X, tmp.Y, tmp.Z));
            }

        }

        /// <summary>
        /// Initialize CellSize for BlockMesh.
        /// </summary>
        private void InitDefaultCellSize()
        {
            double scalarRes = Exporter.Instance.settings.BlockMeshResolution;
            //if (scalarRes < 1)
            //    scalarRes = 1;
            Settings s = Exporter.Instance.settings;
            if (s.DomainX.IsZeroLength())
            {
                m_CellSize.X = Math.Round(m_VecUpperEdgeRight.X - m_VecLowerEdgeLeft.X) * scalarRes;
                m_CellSize.Y = Math.Round(m_VecUpperEdgeRight.Y - m_VecLowerEdgeLeft.Y) * scalarRes;
                m_CellSize.Z = Math.Round(m_VecUpperEdgeRight.Z - m_VecLowerEdgeLeft.Z) * scalarRes;
            }
            else
            {
                m_CellSize.X = Math.Round(s.DomainX.GetLength() * scalarRes);
                m_CellSize.Y = Math.Round(s.DomainY.GetLength() * scalarRes);
                m_CellSize.Z = Math.Round(s.DomainZ.GetLength() * scalarRes);
            }
        }

        /// <summary>
        /// Initialize block dictionary.
        /// </summary>
        private void InitBlocks()
        {
            m_Blocks.Add("hex (0 1 2 3 4 5 6 7) (" + m_CellSize.ToString().Replace(';', ' ') + ")");
            m_Blocks.Add("simpleGrading (" + m_SimpleGrading.ToString().Replace(';', ' ') + ")");
        }

        /// <summary>
        /// Initialize edges dictionary.
        /// </summary>
        private void InitEdges()
        {
            //TO-DO: implement later
        }

        /// <summary>
        /// Initialize boundaries for the blockMesh.
        /// </summary>
        private void InitBoundary()
        {
            Settings s = Exporter.Instance.settings;
            if (s.DomainX.IsZeroLength()) // no ComputationalDomain Family instance
            {
                Dictionary<string, object> boundingBox = new Dictionary<string, object>()
            {
                {"type", "wall"} ,
                {"faces", new ArrayList {
                          {new int[]{ 0, 3, 2, 1 } },
                          {new int[]{ 4, 5, 6, 7 } },
                          {new int[]{ 1, 2, 6, 5 } },
                          {new int[]{ 3, 0, 4, 7 } },
                          {new int[]{ 0, 1, 5, 4 } },
                          {new int[]{ 2, 3, 7, 6 } } }
                }
            };
                m_Boundary.Add(new KeyValuePair<string, object>("boundingBox", boundingBox));
            }
            else
            {

                Dictionary<string, object> frontAndBack = new Dictionary<string, object>()
                {
                    {"type", "wall"} ,
                    {"faces", new ArrayList {
                              {new int[]{ 1, 2, 6, 5 } },
                              {new int[]{ 3, 0, 4, 7 } }}
                    }
                };
                m_Boundary.Add(new KeyValuePair<string, object>("frontAndBack", frontAndBack));

                Dictionary<string, object> inlet = new Dictionary<string, object>()
                {
                    {"type", "patch"} ,
                    {"faces", new ArrayList {
                              {new int[]{ 0, 1, 5, 4 } }}
                    }
                };
                m_Boundary.Add(new KeyValuePair<string, object>("inlet", inlet));
                Dictionary<string, object> outlet = new Dictionary<string, object>()
                {
                    {"type", "patch"} ,
                    {"faces", new ArrayList {
                              {new int[]{ 2, 3, 7, 6 } }}
                    }
                };
                m_Boundary.Add(new KeyValuePair<string, object>("outlet", outlet));
                Dictionary<string, object> lowerWall = new Dictionary<string, object>()
                {
                    {"type", "wall"} ,
                    {"faces", new ArrayList {
                              {new int[]{ 0, 3, 2, 1 } }}
                    }
                };
                m_Boundary.Add(new KeyValuePair<string, object>("lowerWall", lowerWall));
                Dictionary<string, object> upperWall = new Dictionary<string, object>()
                {
                    {"type", "wall"} ,
                    {"faces", new ArrayList {
                              {new int[]{ 4, 5, 6, 7 } }}
                    }
                };
                m_Boundary.Add(new KeyValuePair<string, object>("upperWall", upperWall));
            }
        }

        /// <summary>
        /// Initialize MergePathPair-Dictionary.
        /// </summary>
        private void InitMergePatchPair()
        {
            //TO-DO: implement later.
        }

        /// <summary>
        /// Enlarge vectors which used for creating the Boundingbox in BlockMeshDict.
        /// </summary>
        /// <param name="add">Additional size.</param>
        private void EnlargeBoundingboxVector(float add)
        {
            m_VecLowerEdgeLeft.X -= add;
            m_VecLowerEdgeLeft.Y -= add;
            m_VecLowerEdgeLeft.Z -= add;
            m_VecUpperEdgeRight.X += add;
            m_VecUpperEdgeRight.Y += add;
            m_VecUpperEdgeRight.Z += add;
        }
    }
}
