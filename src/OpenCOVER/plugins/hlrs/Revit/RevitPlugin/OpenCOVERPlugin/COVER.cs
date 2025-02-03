using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Linq;
using System.IO;
using System.Windows.Media.Imaging;
using Autodesk.Revit.ApplicationServices;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using Autodesk.Revit.DB.IFC;
using Autodesk.Revit.UI;
using Autodesk.Revit.UI.Selection;
using Autodesk.Revit.DB.Structure;
using Autodesk.Revit.DB.Events;
using Autodesk.Revit.DB.Visual;
using Bitmap = System.Drawing.Bitmap;
using BoundarySegment = Autodesk.Revit.DB.BoundarySegment;
using ComponentManager = Autodesk.Windows.ComponentManager;
using IWin32Window = System.Windows.Forms.IWin32Window;
using Autodesk.Windows;
using TaskDialog = Autodesk.Revit.UI.TaskDialog;
using Autodesk.Revit.DB.ExtensibleStorage;
using System.Windows.Forms;
using View = Autodesk.Revit.DB.View;
using Panel = Autodesk.Revit.DB.Panel;
using System.Security.Cryptography;
using Autodesk.Revit.Creation;
using Document = Autodesk.Revit.DB.Document;
//using System.Windows.Media.Media3D;

namespace OpenCOVERPlugin
{

    public class COVERMessage
    {

        public MessageBuffer message;
        public int messageType;
        public COVERMessage(MessageBuffer m, int mt)
        {
            message = m;
            messageType = mt;
        }
    }
    public class AxisInfo
    {
        public enum AxisType { Rot = 0, Trans = 1, Scale = 2 };

        public XYZ origin;
        public XYZ direction;
        public int level;
        public double min;
        public double max;
        public AxisType type;
    }
    public class TextureInfo
    {

        public String textuerPath;
        public double sx;
        public double sy;
        public double su;
        public double sv;
        public double ox;
        public double oy;
        public double angle;
        public Color color;
        public double amount;
        public TextureInfo()
        {
            textuerPath = "";
            sx = sy = ox = oy = angle = 0.0;
            amount = 1.0;
            color = new Color(255, 255, 255);
        }
    }
    public class cDesignOptionSet
    {
        public ElementId ID;
        public String name;
        public List<cDesignOption> designOptions;
        public cDesignOptionSet()
        {
            designOptions = new List<cDesignOption>();
        }
    }
    public class cDesignOption
    {
        public DesignOption des;
        public String name;
        public cDesignOptionSet designOptionSet;
        public bool visible;
    }

    public sealed class COVER
    {

        public enum MessageTypes { NewObject = 500, DeleteObject, ClearAll, UpdateObject, NewGroup, NewTransform, EndGroup, AddView, DeleteElement, NewParameters, SetParameter, NewMaterial, NewPolyMesh, NewInstance, EndInstance, SetTransform, UpdateView, AvatarPosition, RoomInfo, NewAnnotation, ChangeAnnotation, ChangeAnnotationText, NewAnnotationID, Views, SetView, Resend, NewDoorGroup, File, Finished, DocumentInfo, NewPointCloud, NewARMarker, DesignOptionSets, SelectDesignOption, IKInfo, Phases, ViewPhase, AddRoomInfo, ObjectInfo, Flip, SelectType };
        public enum ObjectTypes { Mesh = 1, Curve, Instance, Solid, RenderElement, Polymesh, Inline };
        public enum TextureTypes { Diffuse = 1, Bump };
        private Thread messageThread;

        private TcpClient toCOVER;
        private Options mOptions;
        public View3D View3D;
        public String LinkedFileName = "";
        public String CAVEHost ="visent.hlrs.de";
        public enum CoordSystem { ProjectOrigin=1, ProjectBasePoint, SharedBasePoint, GeoReferenced};
        public CoordSystem currentCoordSystem= CoordSystem.ProjectOrigin;
        public bool doRotate=false;
        public RevitLinkInstance CurrentLink = null;
        public int LinkedDocumentID = 0;
        public List<Document> documentList = null;
        public int DocumentID = 0;
        private Document document;
        private UIControlledApplication cApplication;
        public Queue<COVERMessage> messageQueue;
        public ElementId oldDesignOption = null;
        public FilteredElementCollector coordinateSystems;
        public DataStorage ConnectionInfo = null;
        public Schema ConnectionInfoSchema = null;

        public List<cDesignOptionSet> designOptionSets;
        private DesignOptionModifier.Switcher designoptionMod;

        private Dictionary<ElementId, int> phaseDict;
        public void updateVisibility(Document doc)
        {

            ElementId activeOptId = Autodesk.Revit.DB.DesignOption.GetActiveDesignOptionId(doc);
            foreach (cDesignOptionSet os in designOptionSets)
            {
                foreach (cDesignOption des in os.designOptions)
                {
                    if (des.des.Id == activeOptId) // this optionSet contains the activeOptionId, disable all other DesignOptions in this set
                    {
                        des.visible = true;
                        foreach (cDesignOption designoption in os.designOptions)
                        {
                            if (designoption != des)
                            {
                                designoption.visible = false;
                            }
                        }
                        break;
                    }
                    if (des.des.IsPrimary)
                    {
                        des.visible = true;
                    }
                    else
                    {
                        des.visible = false;
                    }
                }
            }
        }
        public bool isVisible(ElementId id)
        {
            foreach (cDesignOptionSet os in designOptionSets)
            {
                foreach (cDesignOption des in os.designOptions)
                {
                    if (des.des.Id == id)
                        return des.visible;
                }
            }
            return false;
        }


        private EventHandler<Autodesk.Revit.UI.Events.IdlingEventArgs> idlingHandler;
        // DLL imports from user32.dll to set focus to
        // Revit to force it to forward the external event
        // Raise to actually call the external event 
        // Execute.

        /// <summary>
        /// The GetForegroundWindow function returns a 
        /// handle to the foreground window.
        /// </summary>
        [DllImport("user32.dll")]
        static extern IntPtr GetForegroundWindow();

        /// <summary>
        /// Move the window associated with the passed 
        /// handle to the front.
        /// </summary>
        [DllImport("user32.dll")]
        static extern bool SetForegroundWindow(
          IntPtr hWnd);

        ExternalEvent messageEvent;

        class externalMessageHandler : IExternalEventHandler
        {

            /// <summary>
            /// Execute method invoked by Revit via the 
            /// external event as a reaction to a call 
            /// to its Raise method.
            /// </summary>
            public void Execute(UIApplication a)
            {
                // As far as I can tell, the external event 
                // should work fine even when switching between
                // different documents. That, however, remains
                // to be tested in more depth (or at all).

                //Document doc = a.ActiveUIDocument.Document;
                //Debug.Assert( doc.Title.Equals( _doc.Title ),
                //  "oops ... different documents ... test this" );
                UIDocument uidoc = a.ActiveUIDocument;

                while (COVER.Instance.messageQueue.Count > 0)
                {
                    COVERMessage m = COVER.Instance.messageQueue.Dequeue();
                    if ((MessageTypes)m.messageType == MessageTypes.AvatarPosition || (MessageTypes)m.messageType == MessageTypes.SetView || (MessageTypes)m.messageType == MessageTypes.Resend)//read only messages
                    {
                        COVER.Instance.HandleMessage(m.message, m.messageType, a.ActiveUIDocument.Document, uidoc, a);
                    }
                    else
                    {
                        Transaction transaction = new(a.ActiveUIDocument.Document);
                        FailureHandlingOptions failOpt = transaction.GetFailureHandlingOptions();

                        failOpt.SetClearAfterRollback(true);
                        failOpt.SetFailuresPreprocessor(new NoWarningsAndErrors());
                        transaction.SetFailureHandlingOptions(failOpt);
                        if (transaction.Start("changeParameters") == TransactionStatus.Started)
                        {
                            COVER.Instance.HandleMessage(m.message, m.messageType, a.ActiveUIDocument.Document, uidoc, a);
                            if (TransactionStatus.Committed != transaction.Commit())
                            {
                                // Autodesk.Revit.UI.TaskDialog.Show("Failure", "Transaction could not be committed");
                                //an error occured end resolution was cancled thus this change can't be committed.
                                // just ignore it and dont bug the user
                            }
                        }
                    }
                }
            }
            /// <summary>
            /// Required IExternalEventHandler interface 
            /// method returning a descriptive name.
            /// </summary>
            public string GetName()
            {
                return string.Format("COVISE Message Event handler");
            }
        }
        externalMessageHandler handler;
        IntPtr ApplicationWindow;
        FamilyInstance avatarObject;

        public PulldownButton pushButtonConnectToOpenCOVER;
        public string ButtonIconsFolder = "";
        BitmapImage connectedImage;
        BitmapImage disconnectedImage;
        Dictionary<ElementId, bool> MaterialInfos;

        COVER()
        {

            try
            {
                CAVEHost = System.Environment.GetEnvironmentVariable("CAVEHOST");
            }
            catch
            {
            }


            documentList = new List<Document>();
            designOptionSets = new List<cDesignOptionSet>();

            phaseDict = new Dictionary<ElementId, int>();

            designoptionMod = new DesignOptionModifier.Switcher();
            mOptions = new Options();
            mOptions.DetailLevel = Autodesk.Revit.DB.ViewDetailLevel.Fine;

            if(View3D!= null)
            {
                mOptions.DetailLevel = View3D.DetailLevel;
            }
            try
            {
                ButtonIconsFolder = System.Environment.GetEnvironmentVariable("COVISEDIR");
            }
            catch
            {
            }
            if (ButtonIconsFolder == null || ButtonIconsFolder.Length == 0)
            {
                ButtonIconsFolder = "c:/src/covise";
            }
            ButtonIconsFolder += "/share/covise/icons";
            connectedImage = new BitmapImage(new Uri(Path.Combine(ButtonIconsFolder, "cover_connected_32.png"), UriKind.Absolute));
            disconnectedImage = new BitmapImage(new Uri(Path.Combine(ButtonIconsFolder, "cover_disconnected_32.png"), UriKind.Absolute));


        }
        /// <summary>
        /// Singleton class which holds the connection to OpenCOVER, is used to communicate with OpenCOVER
        /// </summary>
        public static COVER Instance
        {
            get
            {
                return Nested.instance;
            }
        }
        public bool isConnected()
        {
            if (toCOVER != null && toCOVER.Connected == true)
            {
                return true;
            }
            return false;
        }

        public void setConnected(bool connected)
        {
            if (connected)
            {
                pushButtonConnectToOpenCOVER.LargeImage = connectedImage;
                cApplication.Idling += idlingHandler;
            }
            else
            {
                pushButtonConnectToOpenCOVER.LargeImage = disconnectedImage;
            }

        }
        private Level getLevel(Document document, double height)
        {
            int levelNumber = 0;
            Level lastLevel = null;
            FilteredElementCollector collector = new(document);
            ICollection<Element> collection = collector.OfClass(typeof(Level)).ToElements();
            List<Level> levels = collection.Cast<Level>().ToList();
            levels.Sort((left, right) => left.Elevation.CompareTo(right.Elevation));
            foreach (Level level in levels)
            {
                if (null != level)
                {
                    // keep track of number of levels
                    levelNumber++;
                    if (lastLevel != null)
                    {
                        if (height < level.Elevation && height > lastLevel.Elevation)
                        {
                            return lastLevel;
                        }
                        else if(levelNumber == levels.Count() && height > level.Elevation)
                        {
                            return level;
                        }
                    }
                    lastLevel = level;
                }
            }
            return null;
        }

        private Room getRoom(Document document, double x, double y, double height)
        {
            FilteredElementCollector a = new FilteredElementCollector(document).OfClass(typeof(SpatialElement));

            foreach (SpatialElement e in a)
            {
                Room room = e as Room;

                if (null != room)
                {
                    BoundingBoxXYZ bb = room.get_BoundingBox(null);
                    if (bb != null && (bb.Min.X < x && x < bb.Max.X) && (bb.Min.Y < y && y < bb.Max.Y) && (bb.Min.Z < height && height < bb.Max.Z))
                    {
                        /* SpatialElementBoundaryOptions options = new SpatialElementBoundaryOptions();
                         options.SpatialElementBoundaryLocation = SpatialElementBoundaryLocation.Finish;
                         foreach (IList<Autodesk.Revit.DB.BoundarySegment> boundSegList in room.GetBoundarySegments(options))
                         {
                             boundSegList.
                             foreach (Autodesk.Revit.DB.BoundarySegment boundSeg in boundSegList)
                             {
                                 Element e = boundSeg.Element;
                                 Wall wall = e as Wall;
                                 LocationCurve locationCurve = wall.Location as LocationCurve;
                                 Curve curve = locationCurve.Curve;
                                 roomElementInfo += e.Name + " " + curve.Length + "\n";
                             }
                         }*/
                        return room;
                    }
                }
            }
            return null;
        }
        public static uint SwapUnsignedInt(uint source)
        {
            return (uint)((((source & 0x000000FF) << 24)
            | ((source & 0x0000FF00) << 8)
            | ((source & 0x00FF0000) >> 8)
            | ((source & 0xFF000000) >> 24)));
        }

        public void SendGeometry(FilteredElementIterator iter, UIDocument uidoc, Document doc)
        {

            // document might have changed so set it again.
            document = doc;
            if (uidoc != null) // this is a child document don't clear
            {
                designOptionSets.Clear();
                MessageBuffer mbc = new();
                if (MaterialInfos == null)
                    MaterialInfos = new Dictionary<ElementId, bool>();
                MaterialInfos.Clear();
                mbc = new MessageBuffer();
                mbc.add(1);
                sendMessage(mbc.buf, MessageTypes.ClearAll);
            }
            //FilteredElementCollector collector = new FilteredElementCollector(doc);
            //collector.WhereElementIsElementType().OfClass(typeof(DesignOption));

            FilteredElementCollector collector
              = new(doc);


            collector.OfCategory(BuiltInCategory.OST_DesignOptions);

            foreach (DesignOption des in collector)
            {
                cDesignOption cdes = new();
                cdes.des = des;
                cdes.name = des.Name;
                cdes.visible = false;
                Parameter para = des.get_Parameter(BuiltInParameter.OPTION_SET_ID);
                if (para != null)
                {
                    bool found = false;
                    ElementId osID = para.AsElementId();
                    foreach (cDesignOptionSet os in designOptionSets)
                    {
                        if (os.ID == osID)
                        {
                            found = true;
                            cdes.designOptionSet = os;
                            os.designOptions.Add(cdes);
                            break;
                        }
                    }
                    if (!found)
                    {
                        cDesignOptionSet os = new();
                        os.designOptions.Add(cdes);
                        Element el = doc.GetElement(osID);
                        Parameter paraName = el.get_Parameter(BuiltInParameter.OPTION_SET_NAME);
                        String OptionSetName = paraName.AsString();
                        os.name = OptionSetName;
                        os.ID = osID;
                        cdes.designOptionSet = os;
                        designOptionSets.Add(os);
                    }
                }
            }

            updateVisibility(doc);
            MessageBuffer mb = new();
            mb.add(DocumentID);
            mb.add(designOptionSets.Count);
            foreach (cDesignOptionSet os in designOptionSets)
            {
                mb.add(os.ID.IntegerValue);
                mb.add(os.name);
                mb.add(os.designOptions.Count);
                foreach (cDesignOption des in os.designOptions)
                {
                    mb.add(des.des.Id.IntegerValue);
                    mb.add(des.des.Name);
                    mb.add(des.visible);
                }
            }
            sendMessage(mb.buf, MessageTypes.DesignOptionSets);

            

            ProjectPosition projectPos = doc.ActiveProjectLocation.GetProjectPosition(XYZ.Zero);
            double ProjectNorthAngle = projectPos.Angle;
            double ProjectHeight = projectPos.Elevation;
            XYZ ProjectBasePoint = XYZ.Zero;
            XYZ SharedBasePoint = XYZ.Zero;
            coordinateSystems = new FilteredElementCollector(doc).OfClass(typeof(BasePoint));
            foreach (var bp in coordinateSystems)
            {
                BasePoint basePoint = bp as BasePoint;
                
                if (basePoint.Category.BuiltInCategory == BuiltInCategory.OST_SharedBasePoint)
                {
                    SharedBasePoint = basePoint.Position;
                }
                else if(basePoint.Category.BuiltInCategory == BuiltInCategory.OST_ProjectBasePoint)
                {
                    ProjectBasePoint = basePoint.Position;
                }
                else //unknown basePoint
                {
                    
                }
                if (basePoint.IsShared == true)
                {
                    //this is the survey point

                    /*Location svLoc = basePoint.Location;
                    projectSurvpntX = basePoint.get_Parameter(BuiltInParameter.BASEPOINT_EASTWEST_PARAM).AsDouble();
                    projectSurvpntY = basePoint.get_Parameter(BuiltInParameter.BASEPOINT_NORTHSOUTH_PARAM).AsDouble();
                    projectSurvpntZ = basePoint.get_Parameter(BuiltInParameter.BASEPOINT_ELEVATION_PARAM).AsDouble();*/
                }
                else // this is one of the project base points, we just hope that they all share the same height...
                {
                    if(basePoint.get_Parameter(BuiltInParameter.BASEPOINT_ELEVATION_PARAM) != null)
                    ProjectHeight = basePoint.get_Parameter(BuiltInParameter.BASEPOINT_ELEVATION_PARAM).AsDouble(); 
                    else
                        ProjectHeight = basePoint.Position.Z;
                }
            }

            MessageBuffer mbdocinfo = new();
            mbdocinfo.add(doc.PathName);
            if(doRotate)
            { 
                mbdocinfo.add(ProjectNorthAngle);
            }
            else
            {
                mbdocinfo.add((double)0);
            }
            mbdocinfo.add(projectPos.EastWest);
            mbdocinfo.add(projectPos.NorthSouth);
            mbdocinfo.add(projectPos.Elevation);

            double xo = 0;
            double yo = 0;
            double zo = 0;
            int GeoReference = (int)currentCoordSystem;
            if (currentCoordSystem == CoordSystem.GeoReferenced)
            {
                xo = projectPos.EastWest-ProjectBasePoint.X ;
                yo = projectPos.NorthSouth - ProjectBasePoint.Y;
                zo = projectPos.Elevation - ProjectBasePoint.Z;
            }
            else if (currentCoordSystem == CoordSystem.ProjectBasePoint)
            {
                xo = (-ProjectBasePoint.X);
                yo = (-ProjectBasePoint.Y);
                zo = (-ProjectBasePoint.Z);
            }
            else if (currentCoordSystem == CoordSystem.SharedBasePoint)
            {
                xo = (-SharedBasePoint.X);
                yo = (-SharedBasePoint.Y);
                zo = (-SharedBasePoint.Z);
            }

            /*  IEnumerable<Element> instances = new FilteredElementCollector(document).OfClass(typeof(FamilyInstance)).Where(x => x.Name == "OpenCOVER");

              FilteredElementCollector a = new FilteredElementCollector(document).OfClass(typeof(BasePoint));

              foreach (Element e in instances)
              {
                  IList<Parameter> parameters = e.GetParameters("ProjectOffsetX");
                  if (parameters.Count > 0)
                  {
                      xo = parameters[0].AsDouble();
                  }
                  parameters = e.GetParameters("ProjectOffsetY");
                  if (parameters.Count > 0)
                  {
                      yo = parameters[0].AsDouble();
                  }
                  parameters = e.GetParameters("ProjectOffsetZ");
                  if (parameters.Count > 0)
                  {
                      zo = parameters[0].AsDouble();
                  }
                  parameters = e.GetParameters("GeoReference");
                  if (parameters.Count > 0)
                  {
                      GeoReference = parameters[0].AsInteger();
                  }
              }*/
            mbdocinfo.add(xo);
            mbdocinfo.add(yo);
            mbdocinfo.add(zo);
            mbdocinfo.add(GeoReference);

            sendMessage(mbdocinfo.buf, MessageTypes.DocumentInfo);
            MessageBuffer mbPhases = new();
            // Get the phase array which contains all the phases.
            PhaseArray phases = document.Phases;
            mbPhases.add(phases.Size);
            int phaseNum = 0;
            phaseDict.Clear();
            foreach (Phase ii in phases)
            {
                mbPhases.add(ii.Name);
                phaseDict.Add(ii.Id, phaseNum);
                phaseNum++;
            }
            sendMessage(mbPhases.buf, MessageTypes.Phases);

            if (uidoc != null && uidoc.ActiveView is View3D)
            {
                View3D = uidoc.ActiveView as View3D;
            }
            if (View3D == null)
            {
                iter.Reset();
                while (iter.MoveNext())
                {
                    if (iter.Current is View3D)
                    {
                        View3D = iter.Current as View3D;
                        break;
                    }
                    // this one handles Group.
                }
            }
            if (View3D != null)
            {
                Parameter Phase = View3D.GetParameter(ParameterTypeId.ViewPhase);
                Parameter PhaseFilter = View3D.GetParameter(ParameterTypeId.ViewPhaseFilter);
                MessageBuffer mbView = new();
                if (Phase != null)
                {
                    mbView.add(Phase.AsValueString());
                }
                else
                {
                    mbView.add("");
                }
                if (PhaseFilter != null)
                {
                    mbView.add(PhaseFilter.AsValueString());
                }
                else
                {
                    mbView.add("");
                }
                sendMessage(mbView.buf, MessageTypes.ViewPhase);

            }
            iter.Reset();
            while (iter.MoveNext() && toCOVER.Connected)
            {
                Element el = iter.Current as Element;
                if (el.DesignOption == null || isVisible(el.DesignOption.Id))
                {
                    SendElement(el);
                }
                // this one handles Group.
            }
            // send done

            MessageBuffer mbf = new();
            sendMessage(mbf.buf, MessageTypes.Finished);
        }
        public void SendTypeParameters(FilteredElementIterator iter)
        {
            iter.Reset();
            while (iter.MoveNext())
            {
                sendParameters(iter.Current as Element);
            }
        }
        public void deleteElement(ElementId ID)
        {
            MessageBuffer mb = new();
            mb.add(ID.IntegerValue);
            mb.add(DocumentID);
            sendMessage(mb.buf, MessageTypes.DeleteElement);
        }
        public void designOptionsChanged(Document doc, DesignOption des)
        {
            ElementId activeOptId = Autodesk.Revit.DB.DesignOption.GetActiveDesignOptionId(doc);
            if (activeOptId != oldDesignOption)
            {
                oldDesignOption = activeOptId;
                foreach (cDesignOptionSet os in designOptionSets)
                {
                    foreach (cDesignOption des2 in os.designOptions)
                    {
                        if (des2.visible)
                        {
                            FilteredElementCollector designOptionElements = new(doc);

                            designOptionElements.ContainedInDesignOption(des2.des.Id);

                            foreach (Element el in designOptionElements)
                            {
                                deleteElement(el.Id);
                            }
                        }
                    }
                }
                updateVisibility(doc);
                foreach (cDesignOptionSet os in designOptionSets)
                {
                    foreach (cDesignOption des2 in os.designOptions)
                    {
                        if (des2.visible)
                        {
                            FilteredElementCollector designOptionElements = new(doc);

                            designOptionElements.ContainedInDesignOption(des2.des.Id);

                            foreach (Element el in designOptionElements)
                            {
                                OpenCOVERPlugin.COVER.Instance.SendElement(el);
                            }
                        }
                    }
                }

            }
        }

        public void SendGeometry(FilteredElementIterator iter, List<int> IDs)
        {
            iter.Reset();
            while (iter.MoveNext())
            {
                if (iter.Current is View3D)
                {
                    View3D = iter.Current as View3D;
                    break;
                }
                // this one handles Group.
            }
            iter.Reset();
            while (iter.MoveNext())
            {
                Element elem;
                elem = iter.Current as Element;
                foreach (int ID in IDs)   // get the wall type by the name
                {
                    if (ID == elem.Id.IntegerValue)
                    {
                        MessageBuffer mb = new();
                        mb.add(ID);
                        mb.add(DocumentID);
                        sendMessage(mb.buf, MessageTypes.DeleteObject);
                        SendElement(elem);
                        break;
                    }
                }
                // this one handles Group.
            }
        }

        public void sendFamilySymbolParameters(Element elem)
        {
            // iterate element's parameters
            ParameterSet vrps = new();
            foreach (Parameter para in elem.Parameters)
            {
                if (para.Definition.Name != null && para.Definition.Name.Length > 4)
                {
                    if (String.Compare(para.Definition.Name, 0, "coVR", 0, 4, true) == 0)
                    {
                        vrps.Insert(para);
                    }
                }
            }
            if (vrps.Size > 0)
            {
                MessageBuffer mb = new();
                mb.add(elem.Id.IntegerValue);
                mb.add(DocumentID);
                mb.add(elem.Name + "_FamilySymbol");
                mb.add((int)ObjectTypes.Mesh);
                mb.add(false);//doWalk
                addPhases(mb, elem);
                mb.add(false);
                mb.add(0);

                mb.add(getDepthOny(elem));

                mb.add((byte)220); // color
                mb.add((byte)220);
                mb.add((byte)220);
                mb.add((byte)255);
                mb.add(-1); // material ID

                sendMessage(mb.buf, MessageTypes.NewObject);

                mb = new MessageBuffer();
                mb.add(elem.Id.IntegerValue);
                mb.add(DocumentID);
                mb.add(vrps.Size);
                foreach (Parameter para in vrps)
                {
                    mb.add(para.Id.IntegerValue);
                    mb.add(para.Definition.Name);
                    mb.add((int)para.StorageType);
#if REVIT2019 || REVIT2020 || REVIT2021
                    mb.add("Undefined");
#else
                    mb.add(para.Definition.GetDataType().ToString());
#endif
                    switch (para.StorageType)
                    {
                        case Autodesk.Revit.DB.StorageType.Double:
                            mb.add(para.AsDouble());
                            break;
                        case Autodesk.Revit.DB.StorageType.ElementId:
                            //find out the name of the element
                            ElementId id = para.AsElementId();
                            mb.add(id.IntegerValue);
                            mb.add(DocumentID);
                            break;
                        case Autodesk.Revit.DB.StorageType.Integer:
                            mb.add(para.AsInteger());
                            break;
                        case Autodesk.Revit.DB.StorageType.String:
                            mb.add(para.AsString());
                            break;
                        default:
                            mb.add("Unknown Parameter Storage Type");
                            break;
                    }

                }
                sendMessage(mb.buf, MessageTypes.NewParameters);
            }
        }



        public void sendParameters(Element elem)
        {
            // iterate element's parameters
            ParameterSet vrps = new();
            foreach (Parameter para in elem.Parameters)
            {
                if (para.Definition.Name != null && para.Definition.Name.Length > 4)
                {
                    if (String.Compare(para.Definition.Name, 0, "coVR", 0, 4, true) == 0)
                    {
                        vrps.Insert(para);
                    }
                }
            }
            if (vrps.Size > 0)
            {

                MessageBuffer mb = new();
                mb.add(elem.Id.IntegerValue);
                mb.add(DocumentID);
                mb.add(vrps.Size);
                foreach (Parameter para in vrps)
                {
                    mb.add(para.Id.IntegerValue);
                    mb.add(para.Definition.Name);
                    mb.add((int)para.StorageType);
#if REVIT2019 || REVIT2020 || REVIT2021
                    mb.add("Undefined");
#else
                    mb.add(para.Definition.GetDataType().ToString());
#endif
                    switch (para.StorageType)
                    {
                        case Autodesk.Revit.DB.StorageType.Double:
                            mb.add(para.AsDouble());
                            break;
                        case Autodesk.Revit.DB.StorageType.ElementId:
                            //find out the name of the element
                            ElementId id = para.AsElementId();
                            mb.add(id.IntegerValue);
                            break;
                        case Autodesk.Revit.DB.StorageType.Integer:
                            mb.add(para.AsInteger());
                            break;
                        case Autodesk.Revit.DB.StorageType.String:
                            mb.add(para.AsString());
                            break;
                        default:
                            mb.add("Unknown Parameter Storage Type");
                            break;
                    }

                }
                sendMessage(mb.buf, MessageTypes.NewParameters);
            }
        }
        // Note: Some element does not expose geometry, for example, curtain wall and dimension.
        // In case of a curtain wall, try selecting a whole wall by a window/box instead of a single pick.
        // It will then select internal components and be able to display its geometry.
        //
        public void SendElement(Element elem)
        {
            /* if (elem.GetType() == typeof(Autodesk.Revit.DB.Element))
            {
                return;
            }*/
            if (elem is View)
            {
                sendViewpoint(elem);
            }
            else if (elem is Room)
            {
                sendRoom(elem);
            }
            if (elem.IsHidden(View3D))
            {
                return;
            }
            if (elem is Panel)
            {
                Panel p = elem as Panel;
                if (p.Host != null)
                {
                    if (p.Host.IsHidden(View3D))
                    {
                        return;
                    }
                    if (p.Host.Category != null)
                    {
                        if (!p.Host.Category.get_Visible(View3D as View))
                        {
                            return;
                        }
                    }
                }
            }
            if (elem is Mullion)
            {
                Mullion p = elem as Mullion;
                if (p.Host != null)
                {
                    if (p.Host.IsHidden(View3D))
                    {
                        return;
                    }
                    if (p.Host.Category != null)
                    {
                        if (!p.Host.Category.get_Visible(View3D as View))
                        {
                            return;
                        }
                    }
                }
            }
            if (elem is StairsRun)
            {
                StairsRun p = elem as StairsRun;
                if (p.GetStairs() != null)
                {
                    if (p.GetStairs().IsHidden(View3D))
                    {
                        return;
                    }
                    if (p.GetStairs().Category != null)
                    {
                        if (!p.GetStairs().Category.get_Visible(View3D as View))
                        {
                            return;
                        }
                    }
                }
            }
            if (elem is StairsLanding)
            {
                StairsLanding p = elem as StairsLanding;
                if (p.GetStairs() != null)
                {
                    if (p.GetStairs().IsHidden(View3D))
                    {
                        return;
                    }
                    if (p.GetStairs().Category != null)
                    {
                        if (!p.GetStairs().Category.get_Visible(View3D as View))
                        {
                            return;
                        }
                    }
                }
            }
            if (elem is HandRail)
            {
                HandRail h = elem as HandRail;
                Railing p = elem.Document.GetElement(h.HostRailingId) as Railing;

                if (p != null)
                {
                    if (p.IsHidden(View3D))
                    {
                        return;
                    }
                    if (p.Category != null)
                    {
                        if (!p.Category.get_Visible(View3D as View))
                        {
                            return;
                        }
                    }
                    if (p.HasHost)
                    {
                        Element hostElem = elem.Document.GetElement(p.HostId);
                        if (hostElem.IsHidden(View3D))
                        {
                            return;
                        }
                        if (hostElem.Category != null)
                        {
                            if (!hostElem.Category.get_Visible(View3D as View))
                            {
                                return;
                            }
                        }
                    }
                }
            }
            if (elem is Railing)
            {
                Railing p = elem as Railing;
                if (p.HasHost)
                {
                    Element hostElem = elem.Document.GetElement(p.HostId);
                    if (hostElem.IsHidden(View3D))
                    {
                        return;
                    }
                    if (hostElem.Category != null)
                    {
                        if (!hostElem.Category.get_Visible(View3D as View))
                        {
                            return;
                        }
                    }
                }
            }
            if (elem.Category != null)
            {
                //if(elem.Category.CategoryType != CategoryType.Model)
                //{
                try
                {
                    if (!elem.Category.get_Visible(View3D as View))
                    {
                        return;
                    }
                }
                catch
                {

                }
                //}
            }
            if (elem is ImportInstance)
            {
                String name = elem.Name;
            }
            if (elem is TextNote)
            {
                //sendTextNote(elem);
            }
            else if (elem is RevitLinkInstance)
            {
                RevitLinkInstance link = (RevitLinkInstance)elem;
                /*if(!Autodesk.Revit.DB.RevitLinkType.IsLoaded(document,link.Id))
                {
                    link.Load();
                }*/

                Document linkDoc = link.GetLinkDocument();
                if (linkDoc != null)
                {


                    MessageBuffer mb = new();
                    mb.add(elem.Id.IntegerValue);
                    mb.add(DocumentID);
                    mb.add(elem.Name + "__" + elem.UniqueId.ToString());
                    try
                    {
                        mb.add(link.GetTransform().BasisX.Multiply(link.GetTransform().Scale));
                        mb.add(link.GetTransform().BasisY.Multiply(link.GetTransform().Scale));
                        mb.add(link.GetTransform().BasisZ.Multiply(link.GetTransform().Scale));
                        mb.add(link.GetTransform().Origin);
                    }
                    catch (Autodesk.Revit.Exceptions.InvalidOperationException)
                    {
                        mb.add(new XYZ(1, 0, 0));
                        mb.add(new XYZ(0, 1, 0));
                        mb.add(new XYZ(0, 0, 1));
                        mb.add(new XYZ(0, 0, 0));
                    }
                    sendMessage(mb.buf, MessageTypes.NewTransform);
                    FilteredElementCollector collector = new(linkDoc);
                    LinkedFileName = linkDoc.Title;
                    CurrentLink = link;
                    DocumentID = LinkedDocumentID++;
                    documentList.Add(linkDoc);
                    COVER.Instance.SendGeometry(collector.WhereElementIsNotElementType().GetElementIterator(), null, linkDoc);
                    DocumentID = 0;
                    LinkedFileName = "";
                    CurrentLink = null;
                    mb = new MessageBuffer();
                    sendMessage(mb.buf, MessageTypes.EndGroup);

                }

            }
            else if (elem is DesignOption)
            {
                DesignOption des = (DesignOption)elem;


            }
            // if it is a Group. we will need to look at its components.
            else if (elem is Group)
            {

                /* if we add this, the elements of the Group are duplicates
                   Autodesk.Revit.DB.Group @group = (Autodesk.Revit.DB.Group)elem;
                   Autodesk.Revit.DB.ElementArray members = @group.GetMemberIds;


                   MessageBuffer mb = new MessageBuffer();
                   mb.add(elem.Id.IntegerValue);
            mb.add(DocumentID);
                   mb.add(elem.Name);
                   sendMessage(mb.buf, MessageTypes.NewGroup);
                   foreach (Autodesk.Revit.DB.Element elm in members)
                   {
                      SendElement(elm);
                   }
                   mb = new MessageBuffer();
                   sendMessage(mb.buf, MessageTypes.EndGroup);*/
            }

            else
            {


                if (View3D != null)
                {
                    mOptions.DetailLevel = View3D.DetailLevel;
                }
                // not a group. look at the geom data.
                GeometryElement geom = elem.get_Geometry(mOptions);
                if ((geom != null))
                {
                    SendElement(geom, elem);
                }
            }


        }

        private void sendMaterial(Material materialElement, Element elem)
        {

            if(materialElement==null)
            {
                return;
            }
            try
            {
                bool available = MaterialInfos[materialElement.Id];
                if (available)
                {
                    return;
                }
            }
            catch
            {
            }
            // Material not found thus send a material info
            MaterialInfos[materialElement.Id] = true;

            MessageBuffer mb = new();
            mb.add(materialElement.Id.IntegerValue);
            mb.add(DocumentID);
            TextureInfo ti = getTexture(materialElement, elem, TextureTypes.Diffuse);
            if (ti != null)
            {
                mb.add(ti);
            }
            else
            {
                mb.add(new TextureInfo());
            }
            ti = getTexture(materialElement, elem, TextureTypes.Bump);
            if (ti != null)
            {
                mb.add(ti);
                mb.add(ti.amount);
            }
            else
            {
                mb.add(new TextureInfo());
                mb.add(1.0);
            }
            sendMessage(mb.buf, MessageTypes.NewMaterial);

        }


        private TextureInfo getTexture(Material materialElement, Element elem, TextureTypes tt)
        {

            TextureInfo ti = new();
            AppearanceAssetElement materialAsset = elem.Document.GetElement(materialElement.AppearanceAssetId) as AppearanceAssetElement;
            if (materialAsset != null)
            {
                Asset theAsset = materialAsset.GetRenderingAsset();
                List<AssetProperty> assets = new();
                for (int idx = 0; idx < theAsset.Size; idx++)
                {
                    AssetProperty ap = theAsset.Get(idx);
                    assets.Add(ap);
                }
                String TextureName = "_diffuse";
                if (tt == TextureTypes.Bump)
                    TextureName = "_bump_map";
                // order the properties!
                assets = assets.OrderBy(ap => ap.Name).ToList();
                for (int idx = 0; idx < assets.Count; idx++)
                {
                    AssetProperty ap = assets[idx];
                    if (ap.Name == "common_Tint_color")
                    {
                        AssetPropertyDoubleArray4d val = ap as AssetPropertyDoubleArray4d;
                        if (ti.color.Red == 255 && ti.color.Green == 255 && ti.color.Blue == 255)
                        { // color has not been set et, use generic color
                            ti.color = val.GetValueAsColor();
                        }

                    }
                    if (ap.Name == "ceramic_color")
                    {
                        AssetPropertyDoubleArray4d val = ap as AssetPropertyDoubleArray4d;
                        ti.color = val.GetValueAsColor();
                    }
                    if (ap.Name.Length - 12 >= 0)
                    {
                        if (ap.Name.Substring(ap.Name.Length - 12) == "_bump_amount")
                        {
                            AssetPropertyDouble val = ap as AssetPropertyDouble;
                            if (val != null)
                            {
                                ti.amount = val.Value;
                            }
                            else
                            {
                                AssetPropertyFloat valf = ap as AssetPropertyFloat;
                                if (valf != null)
                                {
                                    ti.amount = valf.Value;
                                }
                            }

                        }
                    }
                    if (ap.Name.Length - TextureName.Length >= 0)
                    {
                        String endString = ap.Name.Substring(ap.Name.Length - TextureName.Length);
                        if (endString == TextureName)
                        {
                            getTextureInfo(ap, ti);
                        }
                    }
                    if (tt == TextureTypes.Diffuse && ap.Name.Length > 6 && (ap.Name.Substring(ap.Name.Length - 6) == "_color"))
                    {
                        {
                            getTextureInfo(ap, ti);
                        }
                    }
                    if (tt == TextureTypes.Diffuse && (ap.Name == "opaque_albedo"))
                    {
                        {
                            getTextureInfo(ap, ti);
                        }
                    }
                    if (tt == TextureTypes.Diffuse && (ap.Name == "surface_albedo" && (ti.textuerPath == "")))
                    {
                        {
                            getTextureInfo(ap, ti);
                        }
                    }
                    if (tt == TextureTypes.Bump && (ap.Name == "surface_normal"))
                    {
                        {
                            getTextureInfo(ap, ti);
                        }
                    }
                    if (tt == TextureTypes.Bump && ap.Name.Length > 12 && (ap.Name.Substring(ap.Name.Length - 12) == "_pattern_map"))
                    {
                        {
                            getTextureInfo(ap, ti);
                        }
                    }
                    if (tt == TextureTypes.Bump && (ap.Name == "surface_roughness" && (ti.textuerPath == "")))
                    {
                        {
                            getTextureInfo(ap, ti);
                        }
                    }

                }
            }
            return ti;
        }

        private double getTransparency(Material materialElement, Element elem)
        {

            TextureInfo ti = new();
            AppearanceAssetElement materialAsset = elem.Document.GetElement(materialElement.AppearanceAssetId) as AppearanceAssetElement;
            if (materialAsset != null)
            {
                Asset theAsset = materialAsset.GetRenderingAsset();
                List<AssetProperty> assets = new();
                for (int idx = 0; idx < theAsset.Size; idx++)
                {
                    AssetProperty ap = theAsset.Get(idx);
                    assets.Add(ap);
                }
                assets = assets.OrderBy(ap => ap.Name).ToList();
                for (int idx = 0; idx < assets.Count; idx++)
                {
                    AssetProperty ap = assets[idx];
                    if (ap.Name == "transparent_distance")
                    {
                        AssetPropertyDistance val = ap as AssetPropertyDistance;
                        return (val.Value / 100.0) * 255;
                    }

                }
            }
            return ((100 - (materialElement.Transparency)) / 100.0) * 255;
        }
        private bool getTextureInfo(AssetProperty ap, TextureInfo ti)
        {
            if (ap.NumberOfConnectedProperties > 0)
            {

                IList<AssetProperty> properties = ap.GetAllConnectedProperties();

                foreach (AssetProperty property in properties)
                {
                    if (property is Asset)
                    {
                        // Nested?

                        Asset asset = property as Asset;
                        int size = asset.Size;
                        for (int i = 0; i < size; i++)
                        {
                            AssetProperty subproperty = asset.Get(i);
                            if (subproperty.Name == "unifiedbitmap_Bitmap")
                            {

                                AssetPropertyString sproperty = subproperty as AssetPropertyString;
                                ti.textuerPath = sproperty.Value;
                                //TaskDialog.Show("TextureName2", ap.Name + "." + subproperty.Name + "   " + ti.textuerPath);
                            }
                            if (subproperty.Name == "bumpmap_Bitmap")
                            {

                                AssetPropertyString sproperty = subproperty as AssetPropertyString;
                                ti.textuerPath = sproperty.Value;
                                //TaskDialog.Show("TextureName2", ap.Name + "." + subproperty.Name + "   " + ti.textuerPath);
                            }
                            else if (subproperty.Name == "texture_RealWorldScaleX")
                            {
                                AssetPropertyDistance val = subproperty as AssetPropertyDistance;
#if REVIT2019 || REVIT2020
                                ti.sx = UnitUtils.Convert(val.Value, val.DisplayUnitType, DisplayUnitType.DUT_DECIMAL_FEET);
#else
                                ti.sx = UnitUtils.ConvertFromInternalUnits(val.Value, UnitTypeId.Feet);
#endif
                            }
                            else if (subproperty.Name == "texture_UScale")
                            {
                                AssetPropertyDouble val = subproperty as AssetPropertyDouble;
                                ti.su = val.Value;
                            }
                            else if (subproperty.Name == "texture_VScale")
                            {
                                AssetPropertyDouble val = subproperty as AssetPropertyDouble;
                                ti.sv = val.Value;
                            }
                            else if (subproperty.Name == "texture_RealWorldScaleY")
                            {
                                AssetPropertyDistance val = subproperty as AssetPropertyDistance;
#if REVIT2019 || REVIT2020
                                ti.sy = UnitUtils.Convert(val.Value, val.DisplayUnitType, DisplayUnitType.DUT_DECIMAL_FEET);
#else
                                ti.sy = UnitUtils.ConvertFromInternalUnits(val.Value, UnitTypeId.Feet);
#endif
                            }
                            else if (subproperty.Name == "texture_RealWorldOffsetX")
                            {
                                AssetPropertyDistance val = subproperty as AssetPropertyDistance;
#if REVIT2019 || REVIT2020
                                ti.ox = UnitUtils.Convert(val.Value, val.DisplayUnitType, DisplayUnitType.DUT_DECIMAL_FEET);
#else
                                ti.ox = UnitUtils.ConvertFromInternalUnits(val.Value, UnitTypeId.Feet);
#endif
                            }
                            else if (subproperty.Name == "texture_RealWorldOffsetY")
                            {
                                AssetPropertyDistance val = subproperty as AssetPropertyDistance;
#if REVIT2019 || REVIT2020
                                ti.oy = UnitUtils.Convert(val.Value, val.DisplayUnitType, DisplayUnitType.DUT_DECIMAL_FEET);
#else
                                ti.oy = UnitUtils.ConvertFromInternalUnits(val.Value, UnitTypeId.Feet);
#endif
                            }
                            else if (subproperty.Name == "texture_WAngle")
                            {
                                AssetPropertyDouble val = subproperty as AssetPropertyDouble;
                                ti.angle = val.Value;
                            }
                            // texture_VScale
                        }

                    }
                }
            }
            else
            {
                return false;
            }
            return true;
        }
        private bool getDepthOny(Element elem)
        {

            OverrideGraphicSettings gs;
            // try to get graphic overrides for DepthOnly rendering
            bool depthOnly = false;
            if (elem.Category != null)
            {
                gs = View3D.GetCategoryOverrides(elem.Category.Id);
                if (gs != null)
                {
                    if (gs.Halftone)
                    {
                        depthOnly = true;
                    }
                }
            }
            gs = View3D.GetElementOverrides(elem.Id);
            if (gs != null)
            {
                if (gs.Halftone)
                {
                    depthOnly = true;
                }
            }
            return depthOnly;
        }
        private void sendGeomElement(Element elem, int num, GeometryObject geomObject, bool createGroups, bool doWalk)
        {
            if (geomObject.Visibility == Autodesk.Revit.DB.Visibility.Visible)
            {


                if ((geomObject is Curve))
                {
                    //mb.add((int)ObjectTypes.Curve);
                    //SendCurve(geomObject);
                }
                else if ((elem is SpatialElement))
                {
                    // don't show room volumes
                }
                else if ((geomObject is GeometryInstance))
                {
                    /*if (elem.Category.Name != "{3}")
                    {
                       if (!elem.Category.get_Visible(View3D as Autodesk.Revit.DB.View))
                       {
                          return;
                       }
                    }*/

                    SendInstance(geomObject as GeometryInstance, elem);
                }
                else if ((geomObject is Mesh))
                {

                    string prefix = "";
                    GraphicsStyle graphicsStyle = elem.Document.GetElement(geomObject.GraphicsStyleId) as GraphicsStyle;
                    if (graphicsStyle != null)
                    {
                        prefix = graphicsStyle.Name + '.';
                    }
                    MessageBuffer mb = new();
                    mb.add(elem.Id.IntegerValue);
                    mb.add(DocumentID);
                    mb.add(elem.Name + "_m_" + num.ToString());
                    mb.add((int)ObjectTypes.Mesh);
                    mb.add(doWalk);
                    addPhases(mb, elem);
                    Mesh meshObj = geomObject as Mesh;
                    SendMesh(meshObj, ref mb, true);// TODO get information on whether a mesh is twosided or not

                    mb.add(getDepthOny(elem));
                    ElementId materialID;
                    materialID = meshObj.MaterialElementId;
                    Material materialElement = elem.Document.GetElement(materialID) as Material;
                    if (materialElement != null)
                    {
                        sendMaterial(materialElement, elem);
                        mb.add(materialElement.Color);
                        mb.add((byte)(getTransparency(materialElement, elem)));
                        mb.add(materialElement.Id.IntegerValue); // material ID

                    }
                    else
                    {
                        if (elem.Category != null && ((elem.Category.LineColor.Red != 0) || (elem.Category.LineColor.Green != 0) || (elem.Category.LineColor.Blue != 0))) // if line color is not black, use this instead of just white
                        {
                            mb.add((byte)elem.Category.LineColor.Red); // color
                            mb.add((byte)elem.Category.LineColor.Green);
                            mb.add((byte)elem.Category.LineColor.Blue);
                            mb.add((byte)255);
                            mb.add(-1); // material ID
                        }
                        else
                        {
                            mb.add((byte)250); // color
                            mb.add((byte)250);
                            mb.add((byte)250);
                            mb.add((byte)255);
                            mb.add(-1); // material ID
                        }
                    }
                    sendMessage(mb.buf, MessageTypes.NewObject);
                    if (num == 0)
                        sendParameters(elem);
                }
                else if ((geomObject is Solid))
                {
                    string prefix = "";
                    GraphicsStyle graphicsStyle = elem.Document.GetElement(geomObject.GraphicsStyleId) as GraphicsStyle;
                    if (graphicsStyle != null)
                    {
                        prefix = graphicsStyle.Name + '.';
                    }
                    MessageBuffer mb = new();
                    mb.add(elem.Id.IntegerValue);
                    mb.add(DocumentID);
                    mb.add(elem.Name + "_s_" + num.ToString());
                    FaceArray faces = ((Solid)geomObject).Faces;
                    if (faces.Size != 0)
                    {

                        ElementId materialID;
                        materialID = faces.get_Item(0).MaterialElementId;
                        Material materialElement = elem.Document.GetElement(materialID) as Material;
                        //   if (materialElement == null || materialElement.MaterialClass != "System")
                        // Imported CAD Models have System Material class by default, thus dont sort them out
                        {
                            if (createGroups)
                            {
                                sendMessage(mb.buf, MessageTypes.NewGroup);
                            }
                            SendSolid(prefix, (Solid)geomObject, elem, doWalk);
                            mb = new MessageBuffer();
                            if (createGroups)
                            {
                                sendMessage(mb.buf, MessageTypes.EndGroup);
                            }
                        }
                    }
                    if (num == 0)
                        sendParameters(elem);
                }
                num++;
            }
        }
        private void extendBB(BoundingBoxXYZ bb, BoundingBoxXYZ ebb)
        {
            double X, Y, Z;
            X = bb.Min.X;
            Y = bb.Min.Y;
            Z = bb.Min.Z;
            if (X > ebb.Min.X)
                X = ebb.Min.X;
            if (Y > ebb.Min.Y)
                Y = ebb.Min.Y;
            if (Z > ebb.Min.Z)
                Z = ebb.Min.Z;
            bb.Min = new XYZ(X, Y, Z);
            X = bb.Max.X;
            Y = bb.Max.Y;
            Z = bb.Max.Z;
            if (X < ebb.Max.X)
                X = ebb.Max.X;
            if (Y < ebb.Max.Y)
                Y = ebb.Max.Y;
            if (Z < ebb.Max.Z)
                Z = ebb.Max.Z;
            bb.Max = new XYZ(X, Y, Z);
        }

        private void SendInline(GeometryElement elementGeom, Element elem)
        {
            FamilyInstance fi = elem as FamilyInstance;
            FamilySymbol family = fi.Symbol;
            if (family != null)
            {
                Parameter p = elem.LookupParameter("url");
                if (p != null)
                {
                    MessageBuffer mb = new();
                    mb.add(elem.Id.IntegerValue);
                    mb.add(DocumentID);
                    mb.add(elem.Name);
                    mb.add((int)ObjectTypes.Inline);
                    mb.add(false);//doWalk
                    addPhases(mb, elem);
                    mb.add(p.AsString());
                    mb.add(false); // DepthOnly
                    sendMessage(mb.buf, MessageTypes.NewObject);
                    sendParameters(elem);
                }
            }
        }
        /// <summary>
        /// Return a location for the given element using
        /// its LocationPoint Point property,
        /// LocationCurve start point, whichever 
        /// is available.
        /// </summary>
        /// <param name="p">Return element location point</param>
        /// <param name="e">Revit Element</param>
        /// <returns>True if a location point is available 
        /// for the given element, otherwise false.</returns>
        static public XYZ GetElementLocation(
          Element e)
        {
            XYZ p = XYZ.Zero;
            if (e == null)
                return p;
            Location loc = e.Location;
            if (null != loc)
            {
                LocationPoint lp = loc as LocationPoint;
                if (null != lp)
                {
                    p = lp.Point;
                }
                else
                {
                    LocationCurve lc = loc as LocationCurve;

                    // Debug.Assert(null != lc,
                    //   "expected location to be either point or curve");
                    if (lc != null)
                    {
                        p = lc.Curve.GetEndPoint(0);
                    }
                }
            }
            return p;
        }

        private String getString(FamilyInstance familyInstance, String paramName)
        {
            IList<Parameter> parameters = familyInstance.GetParameters(paramName);
            if (parameters.Count > 0)
            {
                return parameters[0].AsString();
            }
            return "";
        }
        private int getInt(FamilyInstance familyInstance, String paramName)
        {
            IList<Parameter> parameters = familyInstance.GetParameters(paramName);
            if (parameters.Count > 0)
            {
                return parameters[0].AsInteger();
            }
            return -1;
        }
        private double getDouble(FamilyInstance familyInstance, String paramName)
        {
            IList<Parameter> parameters = familyInstance.GetParameters(paramName);
            if (parameters.Count > 0)
            {
                return parameters[0].AsDouble();
            }
            return -1;
        }
        /// <summary>
        /// Draw geometry of element.
        /// </summary>
        /// <param name="elementGeom"></param>
        /// <remarks></remarks>
        private void SendElement(GeometryElement elementGeom, Element elem)
        {




            if (elementGeom == null /*|| elem.CreatedPhaseId != null && elem.CreatedPhaseId.IntegerValue==-1*/)
            {
                return;
            }
            FamilyInstance fi = elem as FamilyInstance;
            if (fi != null && fi.Symbol.Name == "ARMarker")
            {
                int MarkerID = getInt(fi, "MarkerID");
                if (MarkerID > 0)
                {
                    FamilyInstance hfi = fi.Host as FamilyInstance;
                    Transform t = fi.GetTransform();
                    Transform ht;
                    if (hfi != null)
                    {
                        ht = hfi.GetTransform();
                    }
                    else
                    {
                        ht = Transform.Identity;
                    }
                    Double Size = getDouble(fi, "MarkerSize");
                    Double Offset = getDouble(fi, "Offset");
                    Double Angle = getDouble(fi, "Angle");
                    String MarkerType = getString(fi, "MarkerType");
                    XYZ pos = GetElementLocation(elem);
                    XYZ hostPos = GetElementLocation(fi.Host);
                    MessageBuffer mb = new();
                    mb.add(elem.Id.IntegerValue);
                    mb.add(DocumentID);
                    mb.add(elem.Name + "_" + elem.Id.IntegerValue.ToString());
                    mb.add(t.Origin);
                    mb.add(t.BasisX);
                    mb.add(t.BasisY);
                    mb.add(t.BasisZ);
                    mb.add(ht.Origin);
                    mb.add(ht.BasisX);
                    mb.add(ht.BasisY);
                    mb.add(ht.BasisZ);
                    mb.add(MarkerID);
                    mb.add(Offset);
                    mb.add(Angle);
                    if (fi.Host == null)
                        mb.add(0);
                    else
                        mb.add(fi.Host.Id.IntegerValue);
                    mb.add(Size);
                    mb.add(MarkerType);
                    sendMessage(mb.buf, MessageTypes.NewARMarker);
                }
            }
            int num = 0;
            // check if subobject is an instance
            BoundingBoxXYZ bb = new();
            bb.Min = new XYZ(100000, 100000, 100000);
            bb.Max = new XYZ(-100000, -100000, -100000);
            BoundingBoxXYZ bbL = new();
            bbL.Min = new XYZ(100000, 100000, 100000);
            bbL.Max = new XYZ(-100000, -100000, -100000);
            BoundingBoxXYZ bbR = new();
            bbR.Min = new XYZ(100000, 100000, 100000);
            bbR.Max = new XYZ(-100000, -100000, -100000);
            //bool hasStyle = false;
            bool hasIK = false;
            bool doWalk = false;
            IList<Parameter> ps = elem.GetParameters("doWalk");
            if ((ps.Count > 0) && (ps[0] != null))
            {
                doWalk = ps[0].AsInteger() != 0;
            }
            if (fi != null)
            {
                FamilySymbol family = fi.Symbol;
                if (family != null)
                {

                    bool hasGeometry = false;
                    IEnumerator<GeometryObject> Objects = elementGeom.GetEnumerator();
                    if (Objects.MoveNext())
                    {
                        GeometryObject geomObject = Objects.Current;
                        if (!(geomObject is GeometryInstance))
                        {
                            hasGeometry = true;
                        }
                    }
                    String Name = family.FamilyName;
                    if (hasGeometry && Name.Contains("Kinematics"))
                    {
                        // This object contains kinematics information
                        // search for lines representing axis

                        Options geometryOptions = new();
                        geometryOptions.IncludeNonVisibleObjects = true;
                        geometryOptions.ComputeReferences = true;
                        List<AxisInfo> rotationAxis = new();
                        GeometryElement geom = elem.get_Geometry(geometryOptions);
                        if ((geom != null) && (geom.ElementAt(0) is GeometryInstance))
                        {
                            GeometryInstance geomInst = geom.ElementAt(0) as GeometryInstance;
                            if (geomInst != null)
                            {
                                GeometryElement geom2 = geomInst.GetSymbolGeometry();
                                IEnumerator<GeometryObject> lineObjects = geom2.GetEnumerator();
                                while (lineObjects.MoveNext())
                                {
                                    GeometryObject geomObject = lineObjects.Current;
                                    if (geomObject is Line)
                                    {
                                        Line l = geomObject as Line;
                                        GraphicsStyle graphicsStyle = elem.Document.GetElement(geomObject.GraphicsStyleId) as GraphicsStyle;
                                        if (graphicsStyle != null)
                                        {
                                            if (graphicsStyle.Name.StartsWith("Axis") == true)
                                            {
                                                int axisnumber = 0;
                                                int length = 1;
                                                int numberStart = 4;
                                                AxisInfo.AxisType type = AxisInfo.AxisType.Rot;
                                                if (graphicsStyle.Name[numberStart] == 'T')
                                                {
                                                    type = AxisInfo.AxisType.Trans;
                                                    numberStart++;
                                                }
                                                if (graphicsStyle.Name[numberStart] == 'S')
                                                {
                                                    type = AxisInfo.AxisType.Scale;
                                                    numberStart++;
                                                }

                                                if (graphicsStyle.Name.Length > numberStart + 1 && graphicsStyle.Name[numberStart + 1] >= '0' && graphicsStyle.Name[numberStart + 1] <= '9')
                                                    length = 2;
                                                if (Int32.TryParse(graphicsStyle.Name.Substring(numberStart, length), out axisnumber))
                                                {
                                                    // you know that the parsing attempt
                                                    // was successful
                                                    //we found a rotation axis
                                                    AxisInfo ai = new();
                                                    ai.origin = l.Origin;
                                                    ai.direction = l.Direction;
                                                    ai.min = 0;
                                                    ai.max = 0;
                                                    ai.level = axisnumber;
                                                    ai.type = type;
                                                    rotationAxis.Add(ai);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (rotationAxis.Count > 0)
                        {
                            MessageBuffer mb = new();
                            mb.add(elem.Id.IntegerValue);
                            mb.add(DocumentID);
                            mb.add(rotationAxis.Count);

                            foreach (AxisInfo ai in rotationAxis)
                            {
                                mb.add(ai.level);
                                mb.add(ai.origin);
                                mb.add(ai.direction);
                                mb.add(ai.min);
                                mb.add(ai.max);
                                mb.add((int)ai.type);
                            }
                            sendMessage(mb.buf, MessageTypes.IKInfo);
                            hasIK = true;
                        }
                    }
                }
            }
            if (elem.Category != null)
            {
                if (elem.Category.Name == "Legendenkomponenten")
                    return;
                if (elem.Category.Name == "Legend Components")
                    return;
                if (elem.Category.Id.IntegerValue == (int)BuiltInCategory.OST_Stairs)
                {
                    //hasStyle = false;
                    doWalk = true;
                }
                else if (elem.Category.Id.IntegerValue == (int)BuiltInCategory.OST_Topography)
                {
                    doWalk = true;
                }
                else if (elem.Category.Id.IntegerValue == (int)BuiltInCategory.OST_StairsRuns)
                {
                    doWalk = true;
                }
                else if (elem.Category.Id.IntegerValue == (int)BuiltInCategory.OST_StairsLandings)
                {
                    doWalk = true;
                }
                else if (elem.Category.Id.IntegerValue == (int)BuiltInCategory.OST_StairsTrisers)
                {
                    doWalk = true;
                }
                else if (elem.Category.Id.IntegerValue == (int)BuiltInCategory.OST_Ramps)
                {
                    doWalk = true;
                }
                else if (elem.Category.Id.IntegerValue == (int)BuiltInCategory.OST_Floors)
                {
                    doWalk = true;
                }
                else if (elem.Category.Id.IntegerValue == (int)BuiltInCategory.OST_Doors)
                {
                    GeometryObject geomObject = elementGeom.ElementAt(0);
                    
                }
            }

            /* check if this object is just a geometry instance , then the actual door comes later down the hierarchy */
            int numElements=0;
            IEnumerator<GeometryObject> CountObjects = elementGeom.GetEnumerator();
            while (CountObjects.MoveNext())
            {
                numElements++;
            }
            bool isGeometryInstance=false;
            CountObjects.Reset();
            if (numElements == 1)
            {
                CountObjects.MoveNext();
                if(CountObjects.Current is GeometryInstance)
                    isGeometryInstance = true;

            }


            if ((fi != null) && elem.Category.Id.IntegerValue == (int)BuiltInCategory.OST_Doors && !isGeometryInstance)
            {
                bool hasLeft = false;
                bool hasRight = false;
                // doors are sorted into fixed and moving parts
                IEnumerator<GeometryObject> Objects = elementGeom.GetEnumerator();
                while (Objects.MoveNext())
                {
                    GeometryObject geomObject = Objects.Current;

                    GraphicsStyle graphicsStyle = elem.Document.GetElement(geomObject.GraphicsStyleId) as GraphicsStyle;
                    if (graphicsStyle != null)
                    {
                        if (graphicsStyle.Name == "Frame/Mullion" || graphicsStyle.Name == "Rahmen/Pfosten")
                        {
                            sendGeomElement(elem, num, geomObject, false, false);
                        }
                        if (graphicsStyle.Name.Length > 5 && graphicsStyle.Name.Substring(graphicsStyle.Name.Length - 5) == "_Left")
                        {
                            hasLeft = true;
                        }
                        if (graphicsStyle.Name.Length > 6 && graphicsStyle.Name.Substring(graphicsStyle.Name.Length - 6) == "_Right")
                        {
                            hasRight = true;
                        }
                        Solid solid = geomObject as Solid;
                        if (solid != null)
                        {
                            if (graphicsStyle.Name == "Panel"|| graphicsStyle.Name == "Element")
                            {
                                extendBB(bb, solid.GetBoundingBox());
                            }
                            if (graphicsStyle.Name == "Panel_Left")
                            {
                                extendBB(bbL, solid.GetBoundingBox());
                            }
                            if (graphicsStyle.Name == "Panel_Right")
                            {
                                extendBB(bbR, solid.GetBoundingBox());
                            }
                        }
                    }
                    else // no style --> will be treated as moving parts
                    {
                        //sendGeomElement(elem, num, geomObject, false, false);
                    }
                    num++;
                }
                if (bb.Min.X == 100000)
                {
                    bb.Min = new XYZ(-1, -0.01, -1);
                    bb.Max = new XYZ(1, 0.01, 1);
                }
                if (bbL.Min.X == 100000)
                {
                    bbL.Min = new XYZ(-2, -0.01, -1);
                    bbL.Max = new XYZ(0, 0.01, 1);
                }
                if (bbR.Min.X == 100000)
                {
                    bbR.Min = new XYZ(0, -0.01, -1);
                    bbR.Max = new XYZ(2, 0.01, 1);
                }


                XYZ CenterLeft = new(10000, 0, 0);
                XYZ CenterRight = new(10000, 0, 0);

                // try to find arcs with GraphicsStype _Left or _Right
                Options geometryOptionsArc = new();
                geometryOptionsArc.IncludeNonVisibleObjects = true;
                geometryOptionsArc.ComputeReferences = true;
                List<AxisInfo> rotationAxis = new();
                GeometryElement geomArc = elem.get_Geometry(geometryOptionsArc);
                if ((geomArc != null) && (geomArc.ElementAt(0) is GeometryInstance))
                {
                    GeometryInstance geomInst = geomArc.ElementAt(0) as GeometryInstance;
                    if (geomInst != null)
                    {
                        GeometryElement geom2 = geomInst.GetSymbolGeometry();
                        IEnumerator<GeometryObject> arcObjects = geom2.GetEnumerator();
                        while (arcObjects.MoveNext())
                        {
                            GeometryObject geomObject = arcObjects.Current;
                            if (geomObject is Arc)
                            {
                                Arc a = geomObject as Arc;
                                XYZ c = a.Center;
                                GraphicsStyle graphicsStyle = elem.Document.GetElement(geomObject.GraphicsStyleId) as GraphicsStyle;
                                if (graphicsStyle != null)
                                {
                                    if (graphicsStyle.Name.EndsWith("_Left"))
                                        CenterLeft = c;
                                    if (graphicsStyle.Name.EndsWith("_Right"))
                                        CenterRight = c;
                                }
                                if (CenterLeft.X == 10000)
                                {
                                    CenterLeft = c;
                                }
                                if (CenterLeft != c && CenterRight.X == 10000)
                                {
                                    CenterRight = c;
                                }
                            }
                        }
                    }
                }

                if (CenterLeft.X == 10000)
                {
                    // find arcs in floor plan
                    ViewPlan arbitaryFloorPlan = new FilteredElementCollector(elem.Document).OfClass(typeof(ViewPlan)).Cast<ViewPlan>().Where(x => !x.IsTemplate).FirstOrDefault();
                    Options geometryOptions = new();
                    geometryOptions.IncludeNonVisibleObjects = true;
                    geometryOptions.ComputeReferences = true;
                    geometryOptions.View = arbitaryFloorPlan;
                    GeometryElement geom = elem.get_Geometry(geometryOptions);
                    if ((geom != null) && (geom.ElementAt(0) is GeometryInstance))
                    {
                        GeometryInstance geomInst = geom.ElementAt(0) as GeometryInstance;
                        if (geomInst != null)
                        {
                            GeometryElement geom2 = geomInst.GetSymbolGeometry();
                            IEnumerator<GeometryObject> arcObjects = geom2.GetEnumerator();
                            while (arcObjects.MoveNext())
                            {
                                GeometryObject geomObject = arcObjects.Current;
                                if (geomObject is Arc)
                                {
                                    Arc a = geomObject as Arc;
                                    XYZ c = a.Center;

                                    GraphicsStyle graphicsStyle = elem.Document.GetElement(geomObject.GraphicsStyleId) as GraphicsStyle;
                                    if (graphicsStyle != null)
                                    {
                                        if (graphicsStyle.Name.EndsWith("_Left"))
                                            CenterLeft = c;
                                        if (graphicsStyle.Name.EndsWith("_Right"))
                                            CenterRight = c;
                                    }
                                    if (CenterLeft.X == 10000)
                                    {
                                        CenterLeft = c;
                                    }
                                    if (CenterLeft != c && CenterRight.X == 10000)
                                    {
                                        CenterRight = c;
                                    }
                                }
                            }
                        }
                    }
                }
                num = 0;
                if (hasLeft && hasRight)
                {
                    SendDoorPart(elementGeom, elem, fi, bbL, "_Left", CenterLeft);
                    SendDoorPart(elementGeom, elem, fi, bbR, "_Right", CenterRight);
                }
                else if (hasRight)
                {
                    SendDoorPart(elementGeom, elem, fi, bbR, "_Right", CenterRight);
                }
                else
                {
                    SendDoorPart(elementGeom, elem, fi, bb, "", CenterLeft);
                }

            }
            else
            {

                if (hasIK)
                {
                    MessageBuffer mb = new();
                    mb.add(elem.Id.IntegerValue);
                    mb.add(DocumentID);
                    mb.add(elem.Name);
                    sendMessage(mb.buf, MessageTypes.NewGroup);
                }
                IEnumerator<GeometryObject> Objects = elementGeom.GetEnumerator();
                while (Objects.MoveNext())
                {

                    GeometryObject geomObject = Objects.Current;
                    sendGeomElement(elem, num, geomObject, false, doWalk);
                    num++;

                }
                if (hasIK)
                {
                    MessageBuffer mb = new();
                    mb.add(elem.Id.IntegerValue);
                    mb.add(DocumentID);
                    mb.add(elem.Name);
                    sendMessage(mb.buf, MessageTypes.EndGroup);
                }
            }
        }

        private void sendViewpoint(Element elem)
        {
            View view = (View)elem;
            if (view is View3D && view.IsTemplate != true)
            {

                View3D v3d = (View3D)view;
                if (v3d.CanResetCameraTarget())
                {
                    MessageBuffer mb = new();
                    mb.add(elem.Id.IntegerValue);
                    mb.add(DocumentID);
                    if (LinkedFileName != "")
                        mb.add(LinkedFileName + ":" + elem.Name);
                    else
                        mb.add(elem.Name);
                    XYZ tmpPos = v3d.Origin;
                    XYZ tmpDir = v3d.ViewDirection;
                    XYZ tmpUp = v3d.UpDirection;
                    if (CurrentLink != null)
                    {
                        tmpPos = CurrentLink.GetTransform().OfPoint(tmpPos);
                        tmpDir = CurrentLink.GetTransform().OfVector(tmpDir);
                        tmpUp = CurrentLink.GetTransform().OfVector(tmpUp);
                    }
                    mb.add(tmpPos);
                    mb.add(tmpDir);
                    mb.add(tmpUp);
                    sendMessage(mb.buf, MessageTypes.AddView);
                }
            }
            else
            {
            }
        }

        private void sendRoom(Element elem)
        {
            Room room = (Room)elem;
            if (room != null)
            {
                LocationPoint ElementPosPoint = room.Location as LocationPoint;
                if (ElementPosPoint != null)
                {
                    MessageBuffer mb = new();
                    mb.add(elem.Id.IntegerValue);
                    mb.add(DocumentID);
                    mb.add(room.Name);
                    mb.add(room.Area);

                    mb.add(ElementPosPoint.Point);
                    sendMessage(mb.buf, MessageTypes.AddRoomInfo);
                }
            }
        }
        private void sendTextNote(Element elem)
        {
            TextNote tn = (TextNote)elem;
            if (tn is TextNote)
            {
                MessageBuffer mb = new();
                mb.add(elem.Id.IntegerValue);
                mb.add(DocumentID);
                mb.add(tn.Coord);
                mb.add(tn.Text);
                sendMessage(mb.buf, MessageTypes.NewAnnotation);

            }
            else
            {
            }
        }
        private void SendDoorPart(GeometryElement elementGeom, Element elem, FamilyInstance fi, BoundingBoxXYZ bb, String name, XYZ Center)
        {
            int num = 0;
            double travel = 0.0;
            double travelPercent = 0.0;
            MessageBuffer mb = new();
            mb.add(elem.Id.IntegerValue);
            mb.add(DocumentID);
            mb.add("DoorMovingParts" + name + "_" + elem.Name);
            mb.add(fi.HandFlipped);
            mb.add(fi.HandOrientation);
            mb.add(fi.FacingFlipped);
            mb.add(fi.FacingOrientation);
            mb.add(Center);
            FamilySymbol family = fi.Symbol;
            if (family != null)
            {
                int sliding = 0;
                String oper = family.get_Parameter(BuiltInParameter.DOOR_OPERATION_TYPE).AsString();
                if (oper == "SlidingToLeft")
                {
                    sliding = -1;
                }
                else if (oper == "SlidingToRight")
                {
                    sliding = 1;
                }
                else if (oper == "SLIDING_TO_LEFT")
                {
                    sliding = -1;
                }
                else if (oper == "SLIDING_TO_RIGHT")
                {
                    sliding = 1;
                }
                else if (oper != null && (oper.Length > 7 && oper.Substring(oper.Length - 7) == "SLIDING"))
                    {
                    sliding = 1;
                } 
                else
                {
                    IList<Parameter> ps = family.GetParameters("isSliding");
                    if ((ps.Count > 0) && (ps[0] != null))
                    {
                        sliding = ps[0].AsInteger();
                    }
                }
                mb.add(sliding);
                IList<Parameter> pt = family.GetParameters("travel");
                if ((pt.Count > 0) && (pt[0] != null))
                {
                    travel = pt[0].AsDouble();
                }
                //pt = fi.GetParameters("Öffnungsweite Schiebetüre in %");
                ParameterSet para = fi.Parameters;
                foreach (Parameter p in para)
                {
                    if (p.Definition.Name.EndsWith("re in %"))
                    {

                        travelPercent = p.AsInteger();
                    }
                }
            }
            else
            {
                mb.add(0);
            }
            mb.add(bb.Min);
            mb.add(bb.Max);
            mb.add(travel);
            mb.add(travelPercent);
            sendMessage(mb.buf, MessageTypes.NewDoorGroup);
            int namelen = name.Length;

            IEnumerator<GeometryObject> Objects = elementGeom.GetEnumerator();
            while (Objects.MoveNext())
            {
                GeometryObject geomObject = Objects.Current;

                GraphicsStyle graphicsStyle = elem.Document.GetElement(geomObject.GraphicsStyleId) as GraphicsStyle;
                if (graphicsStyle == null || (graphicsStyle.Name != "Frame/Mullion" && graphicsStyle.Name != "Rahmen/Pfosten"))
                {
                    if (namelen == 0 || (graphicsStyle != null && graphicsStyle.Name.Length > namelen && graphicsStyle.Name.Substring(graphicsStyle.Name.Length - namelen) == name))
                    {
                        sendGeomElement(elem, num, geomObject, false, false);
                    }
                }
                num++;
            }
            mb = new MessageBuffer();
            sendMessage(mb.buf, MessageTypes.EndGroup);
        }

        private void SendInstance(GeometryInstance geomInstance, Element elem)
        {

            MessageBuffer mb = new();
            mb.add(elem.Id.IntegerValue);
            mb.add(DocumentID);
            mb.add(elem.Name + "__" + elem.UniqueId.ToString());
            try
            {
                double scale = geomInstance.Transform.Scale;
                if (elem is PointCloudInstance)
                {
                    scale = 1 / 3.28084;
                }
                if (geomInstance.Transform.BasisX.IsUnitLength())
                {
                    mb.add(geomInstance.Transform.BasisX.Multiply(scale));
                    mb.add(geomInstance.Transform.BasisY.Multiply(scale));
                    mb.add(geomInstance.Transform.BasisZ.Multiply(scale));
                }
                else// it is already scaled
                {
                    mb.add(geomInstance.Transform.BasisX);
                    mb.add(geomInstance.Transform.BasisY);
                    mb.add(geomInstance.Transform.BasisZ);
                }
                mb.add(geomInstance.Transform.Origin);
            }
            catch (Autodesk.Revit.Exceptions.InvalidOperationException)
            {
                mb.add(new XYZ(1, 0, 0));
                mb.add(new XYZ(0, 1, 0));
                mb.add(new XYZ(0, 0, 1));
                mb.add(new XYZ(0, 0, 0));
            }
            sendMessage(mb.buf, MessageTypes.NewTransform);
            //Autodesk.Revit.DB.GeometryElement ge = geomInstance.GetInstanceGeometry(geomInstance.Transform);
            GeometryElement ge = geomInstance.GetSymbolGeometry();
            if (elem.Name == "VrmlInline")
            {
                SendInline(ge, elem);
            }
            else
            {
                SendElement(ge, elem);
                if (elem is PointCloudInstance)
                {
                    PointCloudInstance pointcloud = (PointCloudInstance)elem;
                    String n = pointcloud.Name;
                    /*MessageBuffer mb = new MessageBuffer();
                     * mb.add(elem.Id.IntegerValue);
                mb.add(DocumentID);
                    mb.add(elem.Name + "__" + elem.UniqueId.ToString());
                    mb.add(n);
                    sendMessage(mb.buf, MessageTypes.NewPointCloud);*/


                    MessageBuffer mbpc = new();
                    mbpc.add(elem.Id.IntegerValue);
                    mbpc.add(DocumentID);
                    mbpc.add(elem.Name);
                    mbpc.add((int)ObjectTypes.Inline);
                    mbpc.add(false);//doWalk
                    addPhases(mb, elem);
                    mbpc.add(n + ".e57");

                    mbpc.add(getDepthOny(elem));
                    sendMessage(mbpc.buf, MessageTypes.NewObject);
                    sendParameters(elem);

                }
            }

            mb = new MessageBuffer();
            sendMessage(mb.buf, MessageTypes.EndGroup);

        }
        private bool equalMaterial(ref Material m1, Material m2)
        {
            if (m1 == null && m2 == null)
            {
                return true;
            }
            if (m1 != null && m2 != null)
            {
                if (m1.Id.IntegerValue == m2.Id.IntegerValue)
                {
                    return true;
                }
                return false;
            }
            return false;
        }
        private void SendSolid(string prefix, Solid geomSolid, Element elem, bool doWalk)
        {
            if (elem.Name == "")
                return;
            if (elem.Category != null)
            {
                if (elem.Category.CategoryType == Autodesk.Revit.DB.CategoryType.AnalyticalModel)
                    return;
                if (elem.CreatedPhaseId.IntegerValue == -1)
                    return;
                if (elem.Category.Name == "Legendenkomponenten")
                    return;
                if (elem.Category.Name == "Detailelemente")
                    return;
            }
            FaceArray faces = geomSolid.Faces;
            if (faces.Size == 0)
            {
                return;
            }
            Material m = null;
            bool sameMaterial = true;
            int triangles = 0;
            int maintriangles = 0;
            bool twoSided = false;



            /*Autodesk.Revit.DB.WallType wallType = elem.Document.GetElement(elem.GetTypeId()) as Autodesk.Revit.DB.WallType; // get element type
            if (wallType != null)
            {
                if (wallType.Kind == Autodesk.Revit.DB.WallKind.Curtain)
                {
                    //return; // don't display curtain walls, these are probably fassades with bars and Glazing
                }
            }*/
            /* 
             * Autodesk.Revit.DB.ElementId appearanceID = materialElement.AppearanceAssetId;
                     Autodesk.Revit.DB.AppearanceAssetElement ae = elem.Document.GetElement(appearanceID) as Autodesk.Revit.DB.AppearanceAssetElement;
                     Autodesk.Revit.Utility.Asset asset = ae.GetRenderingAsset();
                     Autodesk.Revit.DB.ParameterSet ps = ae.Parameters;
                     for (int i = 0; i < asset.Size; i++)
                     {
                         Autodesk.Revit.Utility.AssetProperty ap = asset[i];
                         string pn = ap.Name;
                         string val = ap.ToString();

                         System.Collections.Generic.IList<string> props = ap.GetConnectedPropertiesNames();
                         foreach (string p in props)
                         {
                             string pName = p;
                         }
                     }
                     foreach (Autodesk.Revit.DB.Parameter p in ae.Parameters)
                     {
                         string pName = p.AsString();
                         string val = p.AsValueString();
                     }
                    System.Collections.Generic.IList<Autodesk.Revit.Utility.AssetProperty> props2 = asset.GetAllConnectedProperties();*/
            ElementId materialID;
            materialID = faces.get_Item(0).MaterialElementId;
            foreach (Face face in faces)
            {
                bool processedThisFace = false;
                if (face.HasRegions)
                {
                    IList<Face> rfaces = face.GetRegions();
                    if (rfaces.Count > 1)
                    {
                        foreach (Face rface in rfaces)
                        {

                            processedThisFace = true;

                            if (m == null)
                            {
                                materialID = rface.MaterialElementId;
                                Material materialElement = elem.Document.GetElement(rface.MaterialElementId) as Material;
                                m = materialElement;
                                twoSided = rface.IsTwoSided;
                            }
                            Mesh rgeomMesh = rface.Triangulate();
                            if (rgeomMesh != null)
                            {
                                triangles += rgeomMesh.NumTriangles;
                                if (materialID != rface.MaterialElementId)
                                {
                                    sameMaterial = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                if (m == null)
                {
                    materialID = face.MaterialElementId;
                    Material materialElement = elem.Document.GetElement(face.MaterialElementId) as Material;
                    m = materialElement;
                    twoSided = face.IsTwoSided;
                }

                Mesh geomMesh = face.Triangulate();
                if (geomMesh != null)
                {
                    if (!processedThisFace)
                    {
                        triangles += geomMesh.NumTriangles;
                    }
                    maintriangles += geomMesh.NumTriangles;
                    if (materialID != face.MaterialElementId)
                    {
                        sameMaterial = false;
                        break;
                    }
                }
            }
            if (triangles == 0)
                return;
            if (sameMaterial)
            {
                MessageBuffer mb = new();
                mb.add(elem.Id.IntegerValue);
                mb.add(DocumentID);
                mb.add(prefix + elem.Name + "_combined");
                mb.add((int)ObjectTypes.Mesh);
                mb.add(doWalk);
                addPhases(mb, elem);
                mb.add(twoSided);
                mb.add(maintriangles);

                int i = 0;

                mb.incsize(3 * 3 * 2 * (sizeof(float) + 1) * geomSolid.Faces.Size);

                foreach (Face face in geomSolid.Faces)
                {
                    Mesh geomMesh = face.Triangulate();
                    if (geomMesh != null)
                    {
                        mb.incsize(3 * 3 * (sizeof(float) + 1) * geomMesh.NumTriangles);
                        for (i = 0; i < geomMesh.NumTriangles; i++)
                        {
                            MeshTriangle triangle = geomMesh.get_Triangle(i);
                            mb.add((float)triangle.get_Vertex(0).X);
                            mb.add((float)triangle.get_Vertex(0).Y);
                            mb.add((float)triangle.get_Vertex(0).Z);
                            mb.add((float)triangle.get_Vertex(1).X);
                            mb.add((float)triangle.get_Vertex(1).Y);
                            mb.add((float)triangle.get_Vertex(1).Z);
                            mb.add((float)triangle.get_Vertex(2).X);
                            mb.add((float)triangle.get_Vertex(2).Y);
                            mb.add((float)triangle.get_Vertex(2).Z);
                        }
                    }
                }

                mb.add(getDepthOny(elem));
                if (m == null)
                {
                    mb.add((byte)220); // color
                    mb.add((byte)220);
                    mb.add((byte)220);
                    mb.add((byte)255);
                    mb.add(-1); // material ID
                }
                else
                {
                    sendMaterial(m, elem);
                    mb.add(m.Color);
                    mb.add((byte)(getTransparency(m, elem)));
                    mb.add(m.Id.IntegerValue);
                    mb.add(DocumentID);
                }
                sendMessage(mb.buf, MessageTypes.NewObject);
            }
            else
            {
                int num = 0;
                foreach (Face face in geomSolid.Faces)
                {
                    bool processedThisFace = false;
                    if (face.HasRegions)
                    {
                        IList<Face> rfaces = face.GetRegions();
                        if (rfaces.Count > 1)
                        {
                            foreach (Face rface in rfaces)
                            {

                                processedThisFace = true;

                                Mesh geomMesh = rface.Triangulate();
                                if (geomMesh != null)
                                {
                                    MessageBuffer mb = new();
                                    mb.add(elem.Id.IntegerValue);
                                    mb.add(DocumentID);
                                    mb.add(prefix + elem.Name + "_f_" + num.ToString());
                                    mb.add((int)ObjectTypes.Mesh);
                                    mb.add(doWalk);
                                    addPhases(mb, elem);

                                    SendMesh(geomMesh, ref mb, rface.IsTwoSided);

                                    mb.add(getDepthOny(elem));
                                    if (rface.MaterialElementId == ElementId.InvalidElementId)
                                    {
                                        mb.add((byte)220); // color
                                        mb.add((byte)220);
                                        mb.add((byte)220);
                                        mb.add((byte)255);
                                        mb.add(-1); // material ID
                                    }
                                    else
                                    {
                                        Material materialElement = elem.Document.GetElement(rface.MaterialElementId) as Material;

                                        sendMaterial(materialElement, elem);
                                        mb.add(materialElement.Color);
                                        mb.add((byte)(getTransparency(materialElement, elem)));
                                        mb.add(materialElement.Id.IntegerValue);
                                        mb.add(DocumentID);
                                    }
                                    sendMessage(mb.buf, MessageTypes.NewObject);
                                }
                                num++;
                            }
                        }
                    }
                    if (!processedThisFace)
                    {
                        Mesh geomMesh = face.Triangulate();
                        if (geomMesh != null)
                        {
                            MessageBuffer mb = new();
                            mb.add(elem.Id.IntegerValue);
                            mb.add(DocumentID);
                            mb.add(prefix + elem.Name + "_f_" + num.ToString());
                            mb.add((int)ObjectTypes.Mesh);
                            mb.add(doWalk);
                            addPhases(mb, elem);

                            SendMesh(geomMesh, ref mb, face.IsTwoSided);
                            mb.add(getDepthOny(elem));
                            Material materialElement =null;

                            if (face.MaterialElementId != ElementId.InvalidElementId)
                            {
                                materialElement = elem.Document.GetElement(face.MaterialElementId) as Material;
                            }
                            if(materialElement!=null)
                            {
                                sendMaterial(materialElement, elem);
                                mb.add(materialElement.Color);
                                mb.add((byte)(getTransparency(materialElement, elem)));
                                mb.add(materialElement.Id.IntegerValue);
                                mb.add(DocumentID);
                            }
                            else
                            {
                                mb.add((byte)220); // color
                                mb.add((byte)220);
                                mb.add((byte)220);
                                mb.add((byte)255);
                                mb.add(-1); // material ID
                            }
                            sendMessage(mb.buf, MessageTypes.NewObject);
                        }
                        num++;
                    }
                }
            }

            //Autodesk.Revit.DB.Edge Edge = default(Autodesk.Revit.DB.Edge);
            //foreach (var Edge in geomSolid.Edges)
            //{
            //   DrawEdge(Edge);
            //}

        }
        /// <summary>
        /// send a mesh to OpenCOVER.
        /// </summary>
        /// <param name="geomMesh"></param>
        /// <remarks></remarks>
        private void SendMesh(Mesh geomMesh, ref MessageBuffer mb, bool twoSided)
        {

            int i = 0;
            mb.add(twoSided);
            mb.add(geomMesh.NumTriangles);
            mb.incsize(3 * 3 * (sizeof(float) + 1) * geomMesh.NumTriangles);

            for (i = 0; i < geomMesh.NumTriangles; i++)
            {
                MeshTriangle triangle = geomMesh.get_Triangle(i);
                mb.add((float)triangle.get_Vertex(0).X);
                mb.add((float)triangle.get_Vertex(0).Y);
                mb.add((float)triangle.get_Vertex(0).Z);
                mb.add((float)triangle.get_Vertex(1).X);
                mb.add((float)triangle.get_Vertex(1).Y);
                mb.add((float)triangle.get_Vertex(1).Z);
                mb.add((float)triangle.get_Vertex(2).X);
                mb.add((float)triangle.get_Vertex(2).Y);
                mb.add((float)triangle.get_Vertex(2).Z);
            }


        }
        public void addPhases(MessageBuffer mb, Element elem)
        {
            int cPhase = 0;
            int dPhase = -1;
            int result;
            if (phaseDict.TryGetValue(elem.CreatedPhaseId, out result))
            {
                cPhase = result;
            }
            if (phaseDict.TryGetValue(elem.DemolishedPhaseId, out result))
            {
                dPhase = result;
            }
            mb.add(cPhase);
            mb.add(dPhase);
        }

        public static Element FindElementByName(
     Document doc,
     Type targetType,
     string targetName)
        {
            return new FilteredElementCollector(doc)
              .OfClass(targetType)
              .FirstOrDefault(
                e => e.Name.Equals(targetName));
        }

        void TestAllOverloads(
  Document doc,
  XYZ startPoint,
  XYZ endPoint,
  FamilySymbol familySymbol)
        {
            StructuralType stNon = StructuralType.NonStructural;
            StructuralType stBeam = StructuralType.Beam;

            Autodesk.Revit.Creation.Document cd
              = doc.Create;

            View view = doc.ActiveView;
            SketchPlane sk = view.SketchPlane;

            Level level = doc.GetElement(view.LevelId) as Level;

            // Create line from user points

            Curve curve = Line.CreateBound(startPoint, endPoint);

            // Create direction vector from user points

            XYZ dirVec = endPoint - startPoint;

            bool done = false;
            int index = 1;
            while (!done)
            {
                FamilyInstance instance = null;

                // Try different insert methods

                try
                {
                    switch (index)
                    {
                        // public FamilyInstance NewFamilyInstance( 
                        //   XYZ location, FamilySymbol symbol, 
                        //   StructuralType structuralType );

                        case 1:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, stNon);
                            break;

                        case 2:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, stBeam);
                            break;

                        // public FamilyInstance NewFamilyInstance( 
                        //   XYZ origin, FamilySymbol symbol, 
                        //   View specView );

                        case 3:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, null);
                            break;

                        case 4:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, view);
                            break;

                        // public FamilyInstance NewFamilyInstance(
                        //   XYZ location, FamilySymbol symbol, 
                        //   Element host, StructuralType structuralType );

                        case 5:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, sk, stNon);
                            break;

                        case 6:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, sk, stBeam);
                            break;

                        // public FamilyInstance NewFamilyInstance(
                        //   XYZ location, FamilySymbol symbol, 
                        //   XYZ referenceDirection, Element host, 
                        //   StructuralType structuralType );

                        case 7:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, dirVec, sk,
                              stNon);
                            break;

                        case 8:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, dirVec, sk,
                              stBeam);
                            break;

                        // public FamilyInstance NewFamilyInstance(
                        //   Curve curve, FamilySymbol symbol, 
                        //   Level level, StructuralType structuralType );

                        case 9:
                            instance = cd.NewFamilyInstance(
                              curve, familySymbol, null, stNon);
                            break;

                        case 10:
                            instance = cd.NewFamilyInstance(
                              curve, familySymbol, null, stBeam);
                            break;

                        case 11:
                            instance = cd.NewFamilyInstance(
                              curve, familySymbol, level, stNon);
                            break;

                        case 12:
                            instance = cd.NewFamilyInstance(
                              curve, familySymbol, level, stBeam);
                            break;

                        // public FamilyInstance NewFamilyInstance(
                        //   XYZ location, FamilySymbol symbol, 
                        //   Level level, StructuralType structuralType );

                        case 13:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, null, stNon);
                            break;

                        case 14:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, null, stBeam);
                            break;

                        case 15:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, level, stNon);
                            break;

                        case 16:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, level, stBeam);
                            break;

                        // public FamilyInstance NewFamilyInstance(
                        //   XYZ location, FamilySymbol symbol, 
                        //   Element host, Level level, 
                        //   StructuralType structuralType );

                        case 17:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, null, stNon);
                            break;

                        case 18:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, null, stBeam);
                            break;

                        case 19:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, sk, stNon);
                            break;

                        case 20:
                            instance = cd.NewFamilyInstance(
                              startPoint, familySymbol, sk, stBeam);
                            break;

                        default:
                            done = true;
                            break;
                    }
                }
                catch
                { }

                // If instance was created, mark with identifier so I can see which instances were created

                /*  if( null != instance )
                  {
                    Parameter param = instance.get_Parameter( "InstanceIndex" );
                    if( null != param )
                    {
                      param.Set( index );
                    }
                  }*/
                index++;
            }
        }

        List<ElementId> _added_element_ids = new();
        void OnDocumentChanged(
        object sender,
        DocumentChangedEventArgs e)
        {
            _added_element_ids.AddRange(
              e.GetAddedElementIds());
        }
        void createAvatar(Document doc, UIDocument uidoc)
        {
            string FamilyName = "RPC Mann";
            Family family = FindElementByName(doc, typeof(Family), FamilyName) as Family;
            if (null == family)
            {

                string libraryPath = "";
                doc.Application.GetLibraryPaths().TryGetValue("Metric Library", out libraryPath);

                //string _family_folder = libraryPath + "/Außenbauteile/RPC 3D-Objekte/";
                string _family_folder = libraryPath + "";

                //string _family_path = _family_folder + "/RPC Mann.rfa";
                string _family_path = _family_folder + "avatar.rfa";




                doc.LoadFamily(_family_path, out family);

            }
            // Determine the family symbol

            FamilySymbol familySymbol = null;
            Material material = null;

            ISet<ElementId> familySymbolIds = family.GetFamilySymbolIds();
            if (familySymbolIds.Count == 0)
            {
            }
            else
            {

                // Get family symbols which is contained in this family
                foreach (ElementId id in familySymbolIds)
                {
                    familySymbol = family.Document.GetElement(id) as FamilySymbol;
                    // Get family symbol name
                    foreach (ElementId materialId in familySymbol.GetMaterialIds(false))
                    {
                        material = familySymbol.Document.GetElement(materialId) as Material;

                        break;
                    }
                    break;
                }
            }
            /* XYZ newPos = new XYZ(0,0,0);
             int hosttype = family.get_Parameter(BuiltInParameter.FAMILY_HOSTING_BEHAVIOR).AsInteger();
             //       TestAllOverloads(doc, newPos, new XYZ(1, 0, 0), familySymbol);
             //       avatarObject = doc.Create.NewFamilyInstance(newPos, familySymbol, Autodesk.Revit.DB.Structure.StructuralType.NonStructural);  

             doc.Application.DocumentChanged
         += new EventHandler<Autodesk.Revit.DB.Events.DocumentChangedEventArgs>(
           OnDocumentChanged);

             _added_element_ids.Clear();

             // PromptForFamilyInstancePlacement cannot 
             // be called inside transaction.

             uidoc.PromptForFamilyInstancePlacement(familySymbol);

             doc.Application.DocumentChanged
               -= new EventHandler<DocumentChangedEventArgs>(
                 OnDocumentChanged);

             // Access the newly placed family instances:

             int n = _added_element_ids.Count();

             avatarObject = doc.GetElement(_added_element_ids[0]) as FamilyInstance;*/

            avatarObject = FindElementByName(doc, typeof(FamilyInstance), "Avatar") as FamilyInstance;

        }

        private void HandleMessage(MessageBuffer buf, int msgType, Document doc, UIDocument uidoc, UIApplication app)
        {

            // create Avatar object if not present
            /* if (avatarObject == null)
             {
                 createAvatar(doc,uidoc);
             }*/
            switch ((MessageTypes)msgType)
            {

                case MessageTypes.Resend:
                    {
                        FilteredElementCollector collector = new(uidoc.Document);
                        View3D = null;
                        LinkedFileName = "";
                        CurrentLink = null;
                        LinkedDocumentID = 0;
                        DocumentID = 0;
                        documentList.Clear();

                        documentList.Add(doc);
                        COVER.Instance.SendGeometry(collector.WhereElementIsNotElementType().GetElementIterator(), uidoc, uidoc.Document);

                        ElementClassFilter FamilyFilter = new(typeof(FamilySymbol));
                        FilteredElementCollector FamilyCollector = new(uidoc.Document);
                        ICollection<Element> AllFamilies = FamilyCollector.WherePasses(FamilyFilter).ToElements();
                        foreach (FamilySymbol Fmly in AllFamilies)
                        {
                            COVER.Instance.sendFamilySymbolParameters(Fmly);
                        }
                    }
                    break;
                case MessageTypes.SetParameter:
                    {
                        int elemID = buf.readInt();
                        int docID = buf.readInt();
                        int paramID = buf.readInt();

                        ElementId id = new(elemID);
                        Element elem = document.GetElement(id);

                        foreach (Parameter para in elem.Parameters)
                        {
                            if (para.Id.IntegerValue == paramID)
                            {

                                switch (para.StorageType)
                                {
                                    case StorageType.Double:
                                        double d = buf.readDouble();
                                        try
                                        {
                                            para.Set(d);
                                        }
                                        catch
                                        {
                                            TaskDialog.Show("Double", "para.Set failed");
                                        }
                                        d = para.AsDouble();
                                        break;
                                    case Autodesk.Revit.DB.StorageType.ElementId:
                                        //find out the name of the element
                                        int tmpid = buf.readInt();
                                        ElementId eleId = new(tmpid);
                                        try
                                        {
                                            para.Set(eleId);
                                        }
                                        catch
                                        {
                                            TaskDialog.Show("Double", "para.Set failed");
                                        }
                                        break;
                                    case StorageType.Integer:
                                        try
                                        {
                                            para.Set(buf.readInt());
                                        }
                                        catch
                                        {
                                            TaskDialog.Show("Double", "para.Set failed");
                                        }
                                        break;
                                    case StorageType.String:
                                        try
                                        {
                                            para.Set(buf.readString());
                                        }
                                        catch
                                        {
                                            TaskDialog.Show("Double", "para.Set failed");
                                        }
                                        break;
                                    default:
                                        try
                                        {
                                            para.SetValueString(buf.readString());
                                        }
                                        catch
                                        {
                                            TaskDialog.Show("Double", "para.Set failed");
                                        }
                                        break;
                                }

                            }
                        }
                    }
                    break;
                case MessageTypes.SetTransform:
                    {
                        int elemID = buf.readInt();
                        int docID = buf.readInt();
                        double x = buf.readDouble();
                        double y = buf.readDouble();
                        double z = buf.readDouble();

                        ElementId id = new(elemID);
                        Element elem = document.GetElement(id);
                        XYZ translationVec = new(x, y, z);
                        LocationCurve ElementPosCurve = elem.Location as LocationCurve;
                        if (ElementPosCurve != null)
                            ElementPosCurve.Move(translationVec);
                        LocationPoint ElementPosPoint = elem.Location as LocationPoint;
                        if (ElementPosPoint != null)
                            ElementPosPoint.Move(translationVec);
                    }
                    break;
                case MessageTypes.ObjectInfo:
                    {
                        int elemID = buf.readInt();
                        int docID = buf.readInt();

                        ElementId id = new(elemID);
                        Document selectedDoc = documentList[docID];
                        Element elem = selectedDoc.GetElement(id);
                        FamilyInstance fi = elem as FamilyInstance;
                        List<ElementId> elements = new();

                        Element typeElem = selectedDoc.GetElement(elem.GetTypeId());
                        elements.Add(id);
                        uidoc.Selection.SetElementIds(elements);

                        MessageBuffer mb = new();
                        mb.add(elemID);
                        mb.add(docID);
                        mb.add(elem.GetType().Name);
                        mb.add(elem.GetTypeId().IntegerValue);
                        mb.add(typeElem.Name);
                        int flip = 0;
                        if (fi != null)
                        {
                            if (fi.CanFlipFacing)
                                flip = 1;
                            if (fi.CanFlipHand)
                                flip += 2;
                        }
                        mb.add(flip);
                        mb.add(elem.GetValidTypes().Count);
                        foreach (ElementId elemTypeID in elem.GetValidTypes())
                        {
                            Element elemType = selectedDoc.GetElement(elemTypeID);
                            FamilySymbol fs = elemType as FamilySymbol;
                            WallType wt = elemType as WallType;
                            FloorType ft = elemType as FloorType;
                            CeilingType ct = elemType as CeilingType;
                            string typeName = "unknownType";
                            string FamilyName = "unknownFamilyType";
                            if (fs != null)
                            {
                                typeName = fs.Name;
                                FamilyName = fs.FamilyName;
                            }
                            else if (wt != null)
                            {
                                typeName = wt.Name;
                                FamilyName = wt.FamilyName;
                            }
                            else if (ct != null)
                            {
                                typeName = ct.Name;
                                FamilyName = ct.FamilyName;
                            }
                            else if (ft != null)
                            {
                                typeName = ft.Name;
                                FamilyName = ft.FamilyName;
                            }
                            mb.add(typeName);
                            mb.add(elemTypeID.IntegerValue);
                            mb.add(FamilyName);
                        }
                        mb.add(elem.Parameters.Size);
                        foreach (Parameter param in elem.Parameters)
                        {
                            mb.add(param.Id.IntegerValue);
                            mb.add(param.Definition.Name);
                            mb.add((int)param.StorageType);
#if REVIT2019 || REVIT2020 || REVIT2021
                    mb.add("Undefined");
#else
                            //mb.add(param.Definition.GetDataType().ToString());
                            mb.add(param.Definition.GetDataType().TypeId);

#endif
                            switch (param.StorageType)
                            {
                                case Autodesk.Revit.DB.StorageType.Double:
                                    mb.add(param.AsDouble());
                                    break;
                                case Autodesk.Revit.DB.StorageType.ElementId:
                                    //find out the name of the element
                                    ElementId eid = param.AsElementId();
                                    mb.add(eid.IntegerValue);
                                    Element ele = selectedDoc.GetElement(eid);
                                    if (ele != null)
                                        mb.add(ele.Name);
                                    else
                                        mb.add("unknown");
                                    break;
                                case Autodesk.Revit.DB.StorageType.Integer:
                                    mb.add(param.AsInteger());
                                    break;
                                case Autodesk.Revit.DB.StorageType.String:
                                    mb.add(param.AsString());
                                    break;
                                default:
                                    mb.add("Unknown Parameter Storage Type");
                                    break;
                            }
                        }
                        sendMessage(mb.buf, MessageTypes.ObjectInfo);
                    }
                    break;
                case MessageTypes.Flip:
                    {
                        int elemID = buf.readInt();
                        int docID = buf.readInt();
                        int orientation = buf.readInt();

                        ElementId id = new(elemID);
                        Document selectedDoc = documentList[docID];
                        Element elem = selectedDoc.GetElement(id);
                        FamilyInstance fi = elem as FamilyInstance;
                        if (orientation == 0)
                            fi.flipHand();
                        else
                            fi.flipFacing();
                    }
                    break;
                case MessageTypes.SelectType:
                    {
                        int elemID = buf.readInt();
                        int docID = buf.readInt();
                        string typeName = buf.readString();
                        int typeID = buf.readInt();


                        ElementId id = new(elemID);
                        Document selectedDoc = documentList[docID];
                        Element elem = selectedDoc.GetElement(id);
                        FamilyInstance fi = elem as FamilyInstance;
                        ElementId tid = new(typeID);
                        elem.ChangeTypeId(tid);
                    }
                    break;
                case MessageTypes.UpdateView:
                    {
                        int elemID = buf.readInt();
                        int docID = buf.readInt();
                        double ex = buf.readDouble();
                        double ey = buf.readDouble();
                        double ez = buf.readDouble();
                        double dx = buf.readDouble();
                        double dy = buf.readDouble();
                        double dz = buf.readDouble();
                        double ux = buf.readDouble();
                        double uy = buf.readDouble();
                        double uz = buf.readDouble();

                        ElementId id = new(elemID);
                        Element elem = document.GetElement(id);
                        View3D v3d = (View3D)elem;
                        ViewOrientation3D ori = new(new XYZ(ex, ey, ez), new XYZ(ux, uy, uz), new XYZ(dx, dy, dz));
                        v3d.SetOrientation(ori);
                    }
                    break;

                case MessageTypes.NewAnnotation:
                    {

                        int labelNumber = buf.readInt();
                        double x = buf.readDouble();
                        double y = buf.readDouble();
                        double z = buf.readDouble();
                        double h = buf.readDouble();
                        double p = buf.readDouble();
                        double r = buf.readDouble();
                        string labelText = buf.readString();

                        XYZ translationVec = new(x, y, z);
                        View view = document.ActiveView;
                        ElementId currentTextTypeId = document.GetDefaultElementTypeId(ElementTypeGroup.TextNoteType);
                        TextNote tn = Autodesk.Revit.DB.TextNote.Create(document, view.Id, translationVec, labelText, currentTextTypeId);
                        // send back revit ID corresponding to this annotationID 
                        // the mapping of annotationIDs to Revit element IDs is done in OpenCOVER
                        MessageBuffer mb = new();
                        mb.add(labelNumber);
                        mb.add(tn.Id.IntegerValue);
                        sendMessage(mb.buf, MessageTypes.NewAnnotationID);

                    }
                    break;
                case MessageTypes.ChangeAnnotation:
                    {

                        int elemID = buf.readInt();
                        double x = buf.readDouble();
                        double y = buf.readDouble();
                        double z = buf.readDouble();
                        double h = buf.readDouble();
                        double p = buf.readDouble();
                        double r = buf.readDouble();

                        ElementId id = new(elemID);
                        Element elem = document.GetElement(id);

                        TextNote tn = elem as TextNote;
                        if (tn != null)
                        {
                            XYZ translationVec = new(x, y, z);
                            tn.Coord = translationVec;
                        }

                    }
                    break;
                case MessageTypes.SetView:
                    {
                        int currentView = buf.readInt();
                        int docID = buf.readInt();

                        List<View3D> views = new(
                new FilteredElementCollector(doc)
                  .OfClass(typeof(View3D))
                  .Cast<View3D>()
                  .Where(v =>
                    v.CanBePrinted && !v.IsTemplate));
                        int n = 0;
                        foreach (View3D v in views)
                        {
                            if (n == currentView)
                            {
                                try
                                {
                                    uidoc.ActiveView = v;
                                }
                                catch (Autodesk.Revit.Exceptions.ArgumentNullException e)
                                {
                                    Console.WriteLine("Exception information: {0}", e);
                                }
                                catch (Autodesk.Revit.Exceptions.ArgumentException e)
                                {
                                    Console.WriteLine("Exception information: {0}", e);
                                }
                                catch (Autodesk.Revit.Exceptions.InvalidOperationException e)
                                {
                                    Console.WriteLine("Exception information: {0}", e);
                                }
                                break;
                            }
                            n++;
                        }
                    }
                    break;
                case MessageTypes.ChangeAnnotationText:
                    {

                        int elemID = buf.readInt();
                        int docID = buf.readInt();
                        string labelText = buf.readString();

                        ElementId id = new(elemID);
                        Element elem = document.GetElement(id);

                        TextNote tn = elem as TextNote;
                        if (tn != null)
                        {
                            tn.Text = labelText;
                        }

                    }
                    break;
                case MessageTypes.File:
                    {
                        int MatID = buf.readInt();
                        string filePathName = buf.readString();
                        string fileName = buf.readString();
                        FileStream f = null;
                        byte[] b = null;
                        int fileSize = 0;
                        // read File and send it
                        try
                        {
                            f = File.OpenRead(filePathName);
                            fileSize = (int)f.Length;
                            b = new byte[fileSize];
                            int size = f.Read(b, 0, fileSize);
                            if (size != fileSize)
                            {
                                Console.WriteLine("could not read all bytes ", size, fileSize);
                                fileSize = 0;
                            }
                        }
                        catch
                        {
                            try
                            {
                                f = File.OpenRead(fileName);
                                fileSize = (int)f.Length;
                                b = new byte[fileSize];
                                int size = f.Read(b, 0, fileSize);
                                if (size != fileSize)
                                {
                                    Console.WriteLine("could not read all bytes ", size, fileSize);
                                    fileSize = 0;
                                }
                            }
                            catch
                            {
                                fileSize = 0;
                            }
                        }
                        MessageBuffer mb = new();
                        mb.add(MatID);
                        mb.add(fileName);
                        if (f != null)
                            mb.add(fileSize);
                        else
                            mb.add((int)0);
                        if (b != null)
                            mb.add(b);
                        sendMessage(mb.buf, MessageTypes.File);
                        if (f != null)
                            f.Close();
                    }
                    break;
                case MessageTypes.SelectDesignOption:
                    {
                        int elemID = buf.readInt();
                        int docID = buf.readInt();

                        ElementId id = new(elemID);
                        Document selectedDoc = documentList[docID];

                        Element elem = selectedDoc.GetElement(id);
                        DesignOption des = (DesignOption)elem;
                        designoptionMod.SetSelection(des.Name);
                    }
                    break;

                case MessageTypes.AvatarPosition:
                    {
                        double ex = buf.readDouble();
                        double ey = buf.readDouble();
                        double ez = buf.readDouble();
                        double dx = buf.readDouble();
                        double dy = buf.readDouble();
                        double dz = buf.readDouble();
                        Level currentLevel = getLevel(doc, ez);
                        string lev = "";
                        if (currentLevel != null)
                        {
                            lev = currentLevel.Name;
                        }
                        Room testRaum = null;
                        Room currentRoom = null;
                        try
                        {
                            XYZ point = new(ex, ey, ez);
                            currentRoom = doc.GetRoomAtPoint(point);
                            if (currentRoom == null && (currentLevel != null))
                            {
                                point = new XYZ(ex, ey, currentLevel.ProjectElevation);
                                currentRoom = doc.GetRoomAtPoint(point);
                                if (currentRoom == null)
                                {
                                    testRaum = getRoom(doc, ex, ey, ez);
                                    currentRoom = testRaum;
                                }

                            }
                        }
                        catch
                        {
                        }
                        if (currentRoom != null)
                        {

                            string nr = currentRoom.Number;
                            string name = currentRoom.Name;
                            double area = currentRoom.Area;
                            MessageBuffer mb = new();
                            mb.add(nr);
                            mb.add(name);
                            mb.add(area);
                            mb.add(lev);
                            sendMessage(mb.buf, MessageTypes.RoomInfo);
                        }
                        else
                        {
                            string nr = "-1";
                            string name = "No Room";
                            double area = 0.0;
                            MessageBuffer mb = new();
                            mb.add(nr);
                            mb.add(name);
                            mb.add(area);
                            mb.add(lev);
                            sendMessage(mb.buf, MessageTypes.RoomInfo);
                        }
                        if (avatarObject != null)
                        {
                            /*
                            Autodesk.Revit.DB.LocationCurve ElementPosCurve = avatarObject.Location as Autodesk.Revit.DB.LocationCurve;
                            Autodesk.Revit.DB.XYZ translationVec = new Autodesk.Revit.DB.XYZ(ex, ey, ez);
                            ElementPosCurve.Move(translationVec);*/
                        }
                    }
                    break;

            }
        }

        public void handleMessages()
        {
            if (toCOVER != null)
            {
                Byte[] data = new Byte[2000];
                while (true)
                {
                    int len = 0;
                    int numZeros = 0;
                    while (len < 16)
                    {
                        int numRead;
                        try
                        {
                            numRead = toCOVER.GetStream().Read(data, len, 16 - len);
                        }
                        catch (IOException)
                        {
                            // probably socket closed
                            setConnected(false);
                            return;
                        }
                        if (numRead == 0)
                        {
                            numZeros++;
                            if (numZeros > 100)
                            {
                                // this socket is probably closed (Read should block)
                                setConnected(false);
                                return;
                            }
                        }
                        len += numRead;
                    }

                    int msgType = BitConverter.ToInt32(data, 2 * 4);
                    int length = BitConverter.ToInt32(data, 3 * 4);
                    length = (int)ByteSwap.swap((uint)length);
                    msgType = (int)ByteSwap.swap((uint)msgType);
                    len = 0;
                    while (len < length)
                    {
                        int numRead;
                        try
                        {
                            numRead = toCOVER.GetStream().Read(data, len, length - len);
                        }
                        catch (IOException)
                        {
                            // probably socket closed
                            setConnected(false);
                            return;
                        }
                        if (numRead == 0)
                        {
                            numZeros++;
                            if (numZeros > 100)
                            {
                                setConnected(false);
                                return;
                            }
                        }
                        len += numRead;
                    }
                    COVERMessage m = new(new MessageBuffer(data), msgType);
                    messageQueue.Enqueue(m);

                    messageEvent.Raise();
                    IntPtr hBefore = GetForegroundWindow();

                    SetForegroundWindow(Autodesk.Windows.ComponentManager.ApplicationWindow);

                    SetForegroundWindow(hBefore);
                }
            }
        }
        public void startup(UIControlledApplication application)
        {
            idlingHandler = new EventHandler<Autodesk.Revit.UI.Events.IdlingEventArgs>(idleUpdate);
            cApplication = application;
        }
        public void shutdown(UIControlledApplication application)
        {
            application.Idling -= idlingHandler;
            cApplication = application;
        }
        public class NoWarningsAndErrors : IFailuresPreprocessor
        {
            public FailureProcessingResult PreprocessFailures(
              FailuresAccessor a)
            {
                // inside event handler, get all warnings

                //IList<FailureMessageAccessor> failures
                //   = a.GetFailureMessages();
                a.DeleteAllWarnings();
                /*foreach (FailureMessageAccessor f in failures)
                {
                    // check failure definition ids 
                    // against ones to dismiss:

                    FailureDefinitionId id
                      = f.GetFailureDefinitionId();

                    if (BuiltInFailures.RoomFailures.RoomNotEnclosed
                      == id)
                    {
                        a.DeleteWarning(f);
                    }
                }*/

                IList<FailureMessageAccessor> failures = a.GetFailureMessages();
                if (failures.Count > 0)
                    return FailureProcessingResult.ProceedWithRollBack;
                else
                    return FailureProcessingResult.Continue;
            }
        }
        public void idleUpdate(object sender, Autodesk.Revit.UI.Events.IdlingEventArgs e)
        {
            UIApplication uiapp = sender as UIApplication;
            if (uiapp.ActiveUIDocument != null && toCOVER != null)
            {
                e.SetRaiseWithoutDelay();

                Document doc = uiapp.ActiveUIDocument.Document;
                UIDocument uidoc = uiapp.ActiveUIDocument;
                if (!toCOVER.Connected)
                    cApplication.Idling -= idlingHandler;

                while (COVER.Instance.messageQueue.Count > 0)
                {
                    COVERMessage m = COVER.Instance.messageQueue.Dequeue();

                    if ((MessageTypes)m.messageType == MessageTypes.AvatarPosition || (MessageTypes)m.messageType == MessageTypes.SetView || (MessageTypes)m.messageType == MessageTypes.Resend)//read only messages
                    {
                        COVER.Instance.HandleMessage(m.message, m.messageType, doc, uidoc, uiapp);
                    }
                    else
                    {
                        Transaction transaction = new(doc);

                        FailureHandlingOptions failOpt = transaction.GetFailureHandlingOptions();

                        failOpt.SetClearAfterRollback(true);
                        failOpt.SetFailuresPreprocessor(new NoWarningsAndErrors());
                        transaction.SetFailureHandlingOptions(failOpt);

                        if (transaction.Start("changeParameters") == TransactionStatus.Started)
                        {
                            COVER.Instance.HandleMessage(m.message, m.messageType, doc, uidoc, uiapp);
                            if (TransactionStatus.Committed != transaction.Commit())
                            {
                                // Autodesk.Revit.UI.TaskDialog.Show("Failure", "Transaction could not be committed");
                                //an error occured end resolution was cancled thus this change can't be committed.
                                // just ignore it and dont bug the user
                            }
                        }

                    }
                }
            }
        }
        internal void newDocument(Document doc)
        {
            // Try to read our user data
            // Retrieve data storage

            FilteredElementCollector collector =
                new FilteredElementCollector(doc);

            ConnectionInfo = (DataStorage)
                collector
                .OfClass(typeof(DataStorage))
                .FirstElement();

            if (ConnectionInfo == null)
            {
                newConnectionInfo(doc);
            }

            // Retrieve entity from the data storage

            if (ConnectionInfoSchema == null)
                CreateSchema();
            Entity createdInfoEntity = ConnectionInfo.GetEntity(
              ConnectionInfoSchema);

            if (createdInfoEntity.IsValid())
            {
                CAVEHost = createdInfoEntity.Get<string>("CAVEHost");
                doRotate = createdInfoEntity.Get<bool>("doRotate");
                currentCoordSystem = (CoordSystem)createdInfoEntity.Get<int>("CoordSystem");
            }

        }
        internal void closeDocument(int docId)
        {
            ConnectionInfo = null;
        }

        private void newConnectionInfo(Document doc)
        {
            using (Transaction t = new Transaction(
     doc, "add ConnectionInfo"))
            {
                t.Start();

                // Create data storage in new document

                ConnectionInfo = DataStorage.Create(doc);
                if(ConnectionInfoSchema == null)
                    CreateSchema();
                Entity entity = new Entity(
                  ConnectionInfoSchema);

                entity.Set("CAVEHost",
                  CAVEHost);

                entity.Set("doRotate",
                  doRotate);

                entity.Set("CoordSystem",
                  (int)currentCoordSystem);

                // Set entity to the data storage element

                ConnectionInfo.SetEntity(entity);

                t.Commit();
            }
            storeConnectionInfo(doc);
        }

        private void CreateSchema()
        {

            // Create entity which store created info
            SchemaBuilder schemaBuilder = new SchemaBuilder(new Guid("720090CB-DA98-40DC-9425-E53F280DA1F0"));
            schemaBuilder.SetReadAccessLevel(AccessLevel.Public); // allow anyone to read the object
            schemaBuilder.SetWriteAccessLevel(AccessLevel.Public); // restrict writing to this vendor only
            schemaBuilder.SetVendorId("HLRS"); // required because of restricted write-access
            schemaBuilder.SetSchemaName("ConnectionInfo");
            FieldBuilder fieldBuilder = schemaBuilder.AddSimpleField("CAVEHost", typeof(String));
            fieldBuilder.SetDocumentation("Host to connect to when connecting to CAVE is selected.");
            FieldBuilder fieldBuilder2 = schemaBuilder.AddSimpleField("doRotate", typeof(bool));
            fieldBuilder2.SetDocumentation("rotate project to true north.");
            FieldBuilder fieldBuilder3 = schemaBuilder.AddSimpleField("CoordSystem", typeof(int));
            fieldBuilder3.SetDocumentation("coordinatesystem to use.");

            ConnectionInfoSchema = schemaBuilder.Finish(); // register the Schema object
        }

        internal void storeConnectionInfo(Document doc)
        {
            if (ConnectionInfo != null)
            {
                using (Transaction t = new Transaction( doc, "change ConnectionInfo"))
                {
                    t.Start();
                    Entity entity = new Entity(
                  ConnectionInfoSchema);

                    entity.Set("CAVEHost",
                      CAVEHost);

                    entity.Set("doRotate",
                      doRotate);

                    entity.Set("CoordSystem",
                      (int)currentCoordSystem);

                    // Set entity to the data storage element

                    ConnectionInfo.SetEntity(entity);
                    t.Commit();
                }
            }
            else { newConnectionInfo(doc); }
        }

        public bool ConnectToOpenCOVER(string host, int port, Document doc)
        {
            document = doc;
            handler = new externalMessageHandler();
            messageEvent = Autodesk.Revit.UI.ExternalEvent.Create(handler);
            messageQueue = new Queue<COVERMessage>();

            Process[] processes = System.Diagnostics.Process.GetProcessesByName("Revit");

            if (0 < processes.Length)
            {
                ApplicationWindow = processes[0].MainWindowHandle;
            }
            try
            {
                if (toCOVER != null)
                {
                    if (messageThread != null)
                    {
                        try { 
                        messageThread.Abort(); // stop reading from the old toCOVER connection
                        }
                        catch
                        {
                            //there will be an exception, thus ignore it
                        }
                    }
                    toCOVER.Close();
                    toCOVER = null;
                    setConnected(false);
                }

                toCOVER = new TcpClient(host, port);
                if (toCOVER.Connected)
                {
                    // Sends data immediately upon calling NetworkStream.Write.
                    toCOVER.NoDelay = true;
                    LingerOption lingerOption = new(false, 0);
                    toCOVER.LingerState = lingerOption;

                    NetworkStream s = toCOVER.GetStream();
                    Byte[] data = new Byte[256];
                    data[0] = 1;
                    try
                    {
                        //toCOVER.ReceiveTimeout = 1000;
                        s.Write(data, 0, 1);
                        //toCOVER.ReceiveTimeout = 10000;
                    }
                    catch (IOException)
                    {
                        // probably socket closed
                        setConnected(false);
                        return false;
                    }

                    int numRead = 0;
                    try
                    {
                        //toCOVER.ReceiveTimeout = 1000;
                        numRead = s.Read(data, 0, 1);
                        //toCOVER.ReceiveTimeout = 10000;
                    }
                    catch (IOException)
                    {
                        // probably socket closed
                        setConnected(false);
                        return false;
                    }

                    List<View3D> views = new(
        new FilteredElementCollector(doc)
          .OfClass(typeof(View3D))
          .Cast<View3D>()
          .Where(v =>
            v.CanBePrinted && !v.IsTemplate));
                    int n = views.Count;

                    if (0 == n)
                    {
                        setConnected(false);
                        return false;
                    }
                    MessageBuffer mb = new();
                    mb.add(n);
                    foreach (View3D v in views)
                    {
                        mb.add(v.Name);
                    }
                    sendMessage(mb.buf, MessageTypes.Views);

                    if (numRead == 1)
                    {
                        setConnected(true);
                        messageThread = new Thread(new ThreadStart(this.handleMessages));

                        // Start the thread
                        messageThread.Start();

                    }

                    return true;
                }
                System.Windows.Forms.MessageBox.Show("Could not connect to OpenCOVER on localhost, port 31821");
            }
            catch
            {

                setConnected(false);
                System.Windows.Forms.MessageBox.Show("Connection error while trying to connect to OpenCOVER on localhost, port 31821");
            }
            return false;

        }

        public void sendMessage(Byte[] messageData, MessageTypes msgType)
        {
            int len = messageData.Length + (4 * 4);
            Byte[] data = new Byte[len];
            ByteSwap.swapTo((uint)msgType, data, 2 * 4);
            ByteSwap.swapTo((uint)messageData.Length, data, 3 * 4);
            messageData.CopyTo(data, 4 * 4);
            if (toCOVER != null)
            {
                toCOVER.GetStream().Write(data, 0, len);
            }
        }


        class Nested
        {
            // Explicit static constructor to tell C# compiler
            // not to mark type as beforefieldinit
            static Nested()
            {
            }
            internal static readonly COVER instance = new();
        }
    }

}
