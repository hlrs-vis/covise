/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

using System;
using System.Text;
using System.IO;
using System.Windows.Forms;
using Autodesk.Revit.DB;

namespace OpenFOAMInterface.BIM
{
    /// <summary>
    /// Base class providing interface to save data to STL file.
    /// </summary>
    public abstract class SaveData : IDisposable
    {
        // stl file name
        protected string m_FileName = string.Empty;
        // file format: Binary or ASCII
        protected SaveFormat m_SaveFormat = SaveFormat.binary;
        // total triangular number in the model
        protected int m_TriangularNumber = 0;

        /// <summary>
        /// Number of triangulars.
        /// </summary>
        public int TriangularNumber
        {
            get { return m_TriangularNumber; }
            set { m_TriangularNumber = value; }
        }

        /// <summary>
        /// Path to file
        /// </summary>
        public string FileName
        {
            get { return m_FileName; }
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="fileName">STL file name.</param>
        /// <param name="format">File format.</param>
        public SaveData(string fileName, SaveFormat format)
        {
            m_FileName = Exporter.Instance.settings.LocalCaseFolder + "\\constant\\triSurface\\" + fileName;
            m_SaveFormat = format;
        }

        /// <summary>
        /// Close file here if user forget it.
        /// </summary>
        public virtual void Dispose()
        {
            CloseFile();
        }

        /// <summary>
        /// Interface to create file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public abstract bool CreateFile();

        /// <summary>
        /// Interface to write one section include normal and vertex.
        /// </summary>
        /// <param name="normal">Facet normal.</param>
        /// <param name="vertexArr">Vertex array.</param>
        /// <returns>True if succeeded, false if failed.</returns>
        public abstract bool WriteSection(Autodesk.Revit.DB.XYZ normal, double[] vertexArr);

        /// <summary>
        /// Interface for writing solid names in stl.
        /// </summary>
        /// <param name="solid"></param>
        /// <param name="init"></param>
        /// <param name="doc"></param>
        /// <param name="elem"></param>
        /// <returns></returns>
        public abstract bool WriteSolidName(string name, Element elem, bool init);
        public abstract bool WriteSolidName(string name, bool init);

        /// <summary>
        /// Add triangular number section.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public abstract bool AddTriangularNumberSection();

        /// <summary>
        /// Close file handle.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public abstract bool CloseFile();
    }

    /// <summary>
    /// Save date to binary stl file.
    /// </summary>   
    public class SaveDataAsBinary : SaveData
    {
        FileStream fileWriteStream = null;
        BinaryWriter binaryWriter = null;

        private Color m_color = null;

        /// <summary>
        /// Color of trangle mesh to export in Binary format.
        /// </summary>
        public Color Color
        {
            set { m_color = value; }
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="fileName">STL file name.</param>
        /// <param name="format">File format.</param>
        public SaveDataAsBinary(string fileName, SaveFormat format)
            : base(fileName, format)
        {
        }
        /// <summary>
        /// Implement interface of creating a file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool CreateFile()
        {
            bool succeed = true;
            try
            {
                FileAttributes fileAttribute = FileAttributes.Normal;
                if (File.Exists(m_FileName))
                {
                    fileAttribute = File.GetAttributes(m_FileName);
                    FileAttributes tempAtt = fileAttribute & FileAttributes.ReadOnly;
                    if (FileAttributes.ReadOnly == tempAtt)
                    {
                        MessageBox.Show(OpenFOAMInterfaceResource.ERR_FILE_READONLY, OpenFOAMInterfaceResource.MESSAGE_BOX_TITLE,
                              MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                        return false;
                    }
                    File.Delete(m_FileName);
                }

                fileWriteStream = new FileStream(m_FileName, FileMode.Create);
                fileAttribute = File.GetAttributes(m_FileName) | fileAttribute;
                File.SetAttributes(m_FileName, fileAttribute);
                binaryWriter = new BinaryWriter(fileWriteStream);

                binaryWriter.BaseStream.Seek(0, SeekOrigin.Begin);

                // write 80 bytes to STL file as the STL file entity name
                // and preserve 4 bytes space for Triangular Number Section
                byte[] entityName = new byte[84];
                entityName[0] = (byte)/*MSG0*/'n';
                entityName[1] = (byte)/*MSG0*/'a';
                entityName[2] = (byte)/*MSG0*/'m';
                entityName[3] = (byte)/*MSG0*/'e';
                for (int i = 4; i < 84; i++)
                {
                    entityName[i] = (byte)/*MSG0*/'\0';
                }
                binaryWriter.Write(entityName);
            }
            catch (Exception e)
            {
                OpenFOAMDialogManager.ShowDialogException(e);
                succeed = false;
            }
            return succeed;
        }

        /// <summary>
        /// Implement interface of closing the file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool CloseFile()
        {
            bool succeed = true;
            if (null != binaryWriter)
            {
                binaryWriter.Close();
                binaryWriter = null;
            }
            if (null != fileWriteStream)
            {
                fileWriteStream.Close();
                fileWriteStream = null;
            }
            return succeed;
        }

        /// <summary>
        /// Implement interface of writing one section include normal and vertex.
        /// </summary>
        /// <param name="normal">Facet normal.</param>
        /// <param name="vertexArr">Vertex array.</param>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool WriteSection(Autodesk.Revit.DB.XYZ normal, double[] vertexArr)
        {
            bool succeed = true;
            try
            {
                // write 3 float numbers to stl file using 12 bytes. 
                for (int j = 0; j < 3; j++)
                {
                    binaryWriter.Write((float)normal[j]);
                }

                for (int i = 0; i < 9; i++)
                {
                    binaryWriter.Write((float)vertexArr[i]);
                }

                // add color to stl file using two bytes.
                if (m_color != null)
                    binaryWriter.Write((ushort)(((m_color.Red) >> 3) | (((m_color.Green) >> 3) << 5) | (((m_color.Blue) >> 3) << 10)));
                else
                {
                    // add two spaces to stl file using two bytes.
                    byte[] anotherSpace = new byte[2];
                    anotherSpace[0] = (byte)/*MSG0*/'\0';
                    anotherSpace[1] = (byte)/*MSG0*/'\0';
                    binaryWriter.Write(anotherSpace);
                }
            }
            catch (Exception e)
            {
                OpenFOAMDialogManager.ShowDialogException(e);
                succeed = false;
            }
            return succeed;
        }

        /// <summary>
        /// Implement interface of adding triangular number section.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool AddTriangularNumberSection()
        {
            bool succeed = true;
            try
            {
                binaryWriter.BaseStream.Seek(80, SeekOrigin.Begin);

                //write the tringle number to the STL file using 4 bytes.
                binaryWriter.Write(m_TriangularNumber);
            }
            catch (Exception e)
            {
                OpenFOAMDialogManager.ShowDialogException(e);
                succeed = false;
            }
            return succeed;
        }

        public override bool WriteSolidName(string name, Element elem, bool init)
        {
            throw new NotImplementedException();
        }

        public override bool WriteSolidName(string name, bool init)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Save data to ASCII stl file.
    /// </summary>
    public class SaveDataAsAscII : SaveData
    {
        StreamWriter stlFile = null;
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="fileName">STL file name.</param>
        /// <param name="format">File format.</param>
        public SaveDataAsAscII(string fileName, SaveFormat format)
            : base(fileName, format)
        {
        }

        /// <summary>
        /// Implement interface of creating a file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool CreateFile()
        {
            bool succeed = true;
            try
            {
                FileAttributes fileAttribute = FileAttributes.Normal;

                if (File.Exists(m_FileName))
                {
                    fileAttribute = File.GetAttributes(m_FileName);
                    FileAttributes tempAtt = fileAttribute & FileAttributes.ReadOnly;
                    if (FileAttributes.ReadOnly == tempAtt)
                    {
                        MessageBox.Show(OpenFOAMInterfaceResource.ERR_FILE_READONLY, OpenFOAMInterfaceResource.MESSAGE_BOX_TITLE,
                              MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                        return false;
                    }
                    File.Delete(m_FileName);
                }

                stlFile = new(m_FileName);
                stlFile.NewLine = "\n";
                fileAttribute = File.GetAttributes(m_FileName) | fileAttribute;
                File.SetAttributes(m_FileName, fileAttribute);

                //stlFile.WriteLine(/*MSG0*/"solid " + m_FileName); //file header
            }
            catch (Exception e)
            {
                OpenFOAMDialogManager.ShowDialogException(e);
                succeed = false;
            }
            return succeed;
        }

        /// <summary>
        /// Implement interface of closing the file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool CloseFile()
        {
            bool succeed = true;
            if (null != stlFile)
            {
                stlFile.Close();
                stlFile = null;
            }
            return succeed;
        }

        /// <summary>
        /// Include the solid name into STL-File.
        /// </summary>
        /// <param name="name">first part of the solid name</param>
        /// <param name="elem">element id is the second part of the solid name</param>
        /// <param name="init">true symbolizes the begin of the solid definition</param>
        /// <returns>if succeed return true</returns>
        public override bool WriteSolidName(string name, Element elem, bool init)
        {
            if (elem == null)
                return false;

            string newName = name.Replace(" ", "_");
            ElementId id = elem.Id;
            if (init)
            {
                stlFile.WriteLine("solid " + newName + "_" + id);
            }
            else
            {
                stlFile.WriteLine("endsolid " + newName + "_" + id);
            }
            return true;
        }

        /// <summary>
        /// Include the solid name into STL-File.
        /// </summary>
        /// <param name="name">solid name</param>
        /// <param name="init">true symbolizes the begin of the solid definition</param>
        /// <returns>if succeed return true</returns>
        public override bool WriteSolidName(string name, bool init)
        {
            string newName = name.Replace(" ", "_");
            if (init)
            {
                stlFile.WriteLine("solid " + newName);
            }
            else
            {
                stlFile.WriteLine("endsolid " + newName);
            }
            return true;
        }

        /// <summary>
        /// Implement interface of writing one section include normal and vertex.
        /// </summary>
        /// <param name="normal">Facet normal.</param>
        /// <param name="vertexArr">Vertex array.</param>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool WriteSection(Autodesk.Revit.DB.XYZ normal, double[] vertexArr)
        {
            bool succeed = true;
            try
            {
                StringBuilder normalSb = new(/*MSG0*/"  facet normal ");
                for (int j = 0; j < 3; j++)
                {
                    //Numberformat should be english in stl
                    string en_double_normal = normal[j].ToString(System.Globalization.CultureInfo.GetCultureInfo("en-US").NumberFormat);
                    normalSb.Append(en_double_normal).Append(/*MSG0*/" ");
                    //normalSb.Append(normal[j]).Append(/*MSG0*/" ");
                }
                stlFile.WriteLine(normalSb);
                stlFile.WriteLine(/*MSG0*/"    outer loop");
                for (int i = 0; i < 3; i++)
                {
                    StringBuilder vertexSb = new(/*MSG0*/"       vertex ");

                    for (int j = 0; j < 3; j++)
                    {
                        //Numberformat should be english in stl
                        string en_double_vertex = vertexArr[i * 3 + j].ToString(System.Globalization.CultureInfo.GetCultureInfo("en-US").NumberFormat);
                        vertexSb.Append(en_double_vertex).Append(/*MSG0*/" ");
                        //vertexSb.Append(vertexArr[i * 3 + j]).Append(/*MSG0*/" ");
                    }

                    stlFile.WriteLine(vertexSb);
                }
                stlFile.WriteLine(/*MSG0*/"    endloop");
                stlFile.WriteLine(/*MSG0*/"  endfacet");
            }
            catch (Exception e)
            {
                OpenFOAMDialogManager.ShowDialogException(e);
                succeed = false;
            }
            return succeed;
        }


        /// <summary>
        /// ASCII doesn't need to add triangular number
        /// </summary>
        public override bool AddTriangularNumberSection()
        {
            // ASCII doesn't need to add triangular number
            throw new NotImplementedException("ASCII doesn't need to add triangular number");
        }
    }
}
