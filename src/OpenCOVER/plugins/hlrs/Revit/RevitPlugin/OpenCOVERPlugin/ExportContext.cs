using System;
using System.IO;
using System.Text;
using System.Xml;
using System.Collections.Generic;
using System.Diagnostics;

using Autodesk.Revit.DB;
using Autodesk.Revit.Utility;

namespace OpenCOVERPlugin
{
    /// <summary>
    /// Out custom export context
    /// </summary>
    /// <remarks>
    /// An instance of this class is given to a custom exporter in Revit,
    /// which will then redirect all rendering output back to method of this class.
    /// </remarks>
    class ExportContext : IExportContext
    {
        // private data members

        private Document m_doc = null;
        private int num;
        private MaterialNode materialNode;

        /// <summary>
        ///  Constructor
        /// </summary>
        /// <param name="document">Document of which view are going to be exported</param>
        /// <param name="filename">Target filename to export to - overwrite if exists</param>
        /// <param name="options">Export options the user chose.</param>
        public ExportContext(Document document)
        {
            // Normally, there should be validation of the arguments,
            // but we'll just store them, for the sake of simplicity.
            // The export does not start yet. It will later when we get
            // the Start method invoked.
            m_doc = document;
        }

        
        #region Implementation of the IExportContext interface methods

        public bool Start()
        {
        
            return true;   // yes, I am ready; please continue
        }

        public void Finish()
        {
           
        }

        public bool IsCanceled()
        {
            return false;  // in this sample, we never cancel
        }

        public RenderNodeAction OnViewBegin(ViewNode node)
        {
            View3D view = m_doc.GetElement(node.ViewId) as View3D;
            if (view != null)
            {
                MessageBuffer mb = new MessageBuffer();
                mb.add(view.Id.IntegerValue);
                mb.add(view.Name);
                mb.add(view.Origin);
                mb.add(view.ViewDirection);
                mb.add(view.UpDirection);
                OpenCOVERPlugin.COVER.Instance.sendMessage(mb.buf, OpenCOVERPlugin.COVER.MessageTypes.AddView);
            }

            // Setting our default LoD for the view
            // The scale goes from 0 to 10, but the value close to the edges
            // aren't really that usable, except maybe of experimenting

            node.LevelOfDetail = 5;   // a good middle ground 
           
            return RenderNodeAction.Proceed;
        }

        public void OnViewEnd(ElementId elementId)
        {
        }

        public RenderNodeAction OnElementBegin(ElementId elementId)
        {
            num++;
           // TODO check if this element is in our changed list.

            // if we proceed, we get everything that belongs to the element
            /*MessageBuffer mb = new MessageBuffer();
            mb.add(elementId.IntegerValue);

            Element elem = m_doc.GetElement(elementId) as Element;
            mb.add(elem.Name + "_m_" + num.ToString());
            num++;
            mb.add((int)OpenCOVERPlugin.COVER.ObjectTypes.Mesh);
            Autodesk.Revit.DB.Mesh meshObj = geomObject as Autodesk.Revit.DB.Mesh;
            SendMesh(meshObj, ref mb, true);// TODO get information on whether a mesh is twosided or not

            Autodesk.Revit.DB.ElementId materialID;
            materialID = meshObj.MaterialElementId;
            Autodesk.Revit.DB.Material materialElement = elem.Document.GetElement(materialID) as Autodesk.Revit.DB.Material;
            if (materialElement != null)
            {
                mb.add(materialElement.Color);
                mb.add((byte)(((100 - (materialElement.Transparency)) / 100.0) * 255));
                mb.add(materialID.IntegerValue); // material ID
            }
            else
            {
                mb.add((byte)250); // color
                mb.add((byte)250);
                mb.add((byte)250);
                mb.add((byte)255);
                mb.add(-1); // material ID
            }
            sendMessage(mb.buf, MessageTypes.NewObject);
            sendParameters(elem);*/
            return RenderNodeAction.Proceed;
        }

        public void OnElementEnd(ElementId elementId)
        {
        }

        public RenderNodeAction OnInstanceBegin(InstanceNode node)
        {
            Transform t = node.GetTransform();
            MessageBuffer mb = new MessageBuffer();

            mb.add(node.GetSymbolId().IntegerValue);
            mb.add(node.NodeName);
            mb.add(t.BasisX.Multiply(t.Scale));
            mb.add(t.BasisY.Multiply(t.Scale));
            mb.add(t.BasisZ.Multiply(t.Scale));
            mb.add(t.Origin);
            OpenCOVERPlugin.COVER.Instance.sendMessage(mb.buf, OpenCOVERPlugin.COVER.MessageTypes.NewTransform);
            return RenderNodeAction.Proceed;
        }

        public void OnInstanceEnd(InstanceNode node)
        {
            MessageBuffer mb = new MessageBuffer();
            OpenCOVERPlugin.COVER.Instance.sendMessage(mb.buf, OpenCOVERPlugin.COVER.MessageTypes.EndGroup);
        }

        public RenderNodeAction OnLinkBegin(LinkNode node)
        {

            num++;
            return RenderNodeAction.Proceed;
        }

        public void OnLinkEnd(LinkNode node)
        {
           
        }

        public RenderNodeAction OnFaceBegin(FaceNode node)
        {
            // remember, the exporter notifies about faces only
            // when it was requested at the time the export process started.
            // If it was not requested, the context would receive tessellated
            // meshes only, but not faces. Otherwise, both woudl be received
            // and it woudl be up to this context here to use what is needed.

            num++;
            return RenderNodeAction.Proceed;
        }

        public void OnFaceEnd(FaceNode node)
        {
            
        }

        public void OnLight(LightNode node)
        {
            // More info about lights can be acquired here using the standard Light API

            return;
        }

        public void OnRPC(RPCNode node)
        {
            // RPCs cannot get (due to copyrights) any info besides assets.
            // There is currently no public API for RPCs

            return;
        }

        public void OnDaylightPortal(DaylightPortalNode node)
        {
            // Like RPCs, Daylight Portals too have their assets available only.
            // THere is no other public API for them currently available.

            
            return;
        }

        public void OnMaterial(MaterialNode node)
        {
            materialNode = node;
            MessageBuffer mb = new MessageBuffer();

            mb.add(node.MaterialId.IntegerValue);
            mb.add(node.NodeName);
            mb.add(node.Color);
            mb.add((byte)(((100 - (node.Transparency)) / 100.0) * 255));
            Asset asset;
            if (node.HasOverriddenAppearance)
            {
                asset = node.GetAppearanceOverride();
            }
            else
            {
                asset = node.GetAppearance();
            }
            String textureName = "";
            AssetProperties properties = asset as AssetProperties;
            for (int index = 0; index < asset.Size; index++)
            {
                
                if (properties[index].Type == AssetPropertyType.APT_Reference)
                {
                    AssetPropertyReference e = properties[index] as AssetPropertyReference;
                    if (e != null)
                    {
                        AssetProperty p = e.GetConnectedProperty(0);
                        if (p.Type == AssetPropertyType.APT_Asset)
                        {
                            Asset a = p as Asset;
                            if (a != null)
                            {
                                Boolean foundValidTexture = false;
                                AssetProperties prop = a as AssetProperties;
                                for (int ind = 0; ind < a.Size; ind++)
                                {
                                    if (prop[ind].Name == "unifiedbitmap_Bitmap")
                                    {

                                        AssetPropertyString ps = prop[ind] as AssetPropertyString;
                                        if (ps.Value != "")
                                        {
                                            textureName = ps.Value;
                                            foundValidTexture = true;
                                        }
                                    }
                                    if (prop[ind].Name == "texture_URepeat")
                                    {
                                        AssetPropertyBoolean ps = prop[ind] as AssetPropertyBoolean;
                                        if (foundValidTexture)
                                        {

                                        }
                                        //textureName = ps.Value;
                                    } 
                                    if (prop[ind].Name == "texture_VRepeat")
                                    {

                                        AssetPropertyBoolean ps = prop[ind] as AssetPropertyBoolean;
                                        if (foundValidTexture)
                                        {

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            mb.add(textureName);
            OpenCOVERPlugin.COVER.Instance.sendMessage(mb.buf, OpenCOVERPlugin.COVER.MessageTypes.NewMaterial);

            return;
        }

        public void OnPolymesh(PolymeshTopology node)
        {

            num++;
            // A polymesh is the very bottom node in the hierarchy of rendering a model.
            // Not every exporter may need need polymeshes; on the other hand, graphic
            // renderer and exporter alike will mostly need nothing but polymeshes.

  /*          m_writer.WriteStartElement("Polymesh");
            m_writer.WriteAttributeString("Facets", node.NumberOfFacets.ToString());
            m_writer.WriteAttributeString("Points", node.NumberOfPoints.ToString());
            m_writer.WriteAttributeString("UVs", node.NumberOfUVs.ToString());
            m_writer.WriteAttributeString("Normals", node.NumberOfNormals.ToString());

            m_writer.WriteElementString("Facets", StringConversions.ValueToString(node.GetFacets()));

            if (m_exportOptions.ApplyTransforms)   // we want either raw or transformed coordinates
            {
                m_writer.WriteElementString("Points", StringConversions.ValueToString(m_transforms.ApplyTransform(node.GetPoints())));
                m_writer.WriteElementString("Normals", StringConversions.ValueToString(m_transforms.ApplyTransform(node.GetNormals())));
            }
            else
            {
                m_writer.WriteElementString("Points", StringConversions.ValueToString(node.GetPoints()));
                m_writer.WriteElementString("Normals", StringConversions.ValueToString(node.GetNormals()));

            }

            if (node.NumberOfUVs > 0)
            {
                m_writer.WriteElementString("UVs", StringConversions.ValueToString(node.GetUVs()));
            }

            m_writer.WriteEndElement();
            */
            return;
        }

        #endregion

    }  // class

}  // namespace
