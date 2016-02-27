using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Office.Interop.Excel;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;

namespace OfficeConsole
{
  public class Excel : BasicOfficeControl
  {
    private Microsoft.Office.Interop.Excel.Application excel = null;
    private _Workbook workbook = null;
    private _Worksheet worksheet = null;

    private Chart activeChart = null;

    // Shortcut for Missing.Value
    private object missing = Missing.Value;

    // Excel //////////////////////////////////////////////////////////////////////////////////////////

    /// <summary>
    /// Start Excel
    /// </summary>
    public void start()
    {
      if (this.excel == null)
      {
        this.excel = new Microsoft.Office.Interop.Excel.Application();
        this.excel.Visible = true;


        this.excel.AfterCalculate += new AppEvents_AfterCalculateEventHandler(afterCalculate);
        this.excel.SheetActivate += new AppEvents_SheetActivateEventHandler(sheetActivate);
        this.excel.SheetCalculate += new AppEvents_SheetCalculateEventHandler(sheetCalculate);
        this.excel.SheetChange += new AppEvents_SheetChangeEventHandler(cellChanged);
        this.excel.SheetSelectionChange += new AppEvents_SheetSelectionChangeEventHandler(sheetSelectionChange);
        this.excel.WindowActivate += new AppEvents_WindowActivateEventHandler(windowActivate);
        this.excel.WindowDeactivate += new AppEvents_WindowDeactivateEventHandler(windowDeactivate);
        this.excel.WorkbookActivate += new AppEvents_WorkbookActivateEventHandler(workbookActivate);
        this.excel.WorkbookDeactivate += new AppEvents_WorkbookDeactivateEventHandler(workbookDeactivate);
        this.excel.WorkbookOpen += new AppEvents_WorkbookOpenEventHandler(workbookOpen);
      }

    }

    public void update()
    {
      if (this.excel != null && this.workbook != null && this.worksheet != null)
      {
        if (this.activeChart != this.excel.ActiveChart)
        {
          if (this.excel.ActiveChart != null)
          {
            if (this.activeChart != null)
              OfficeConsole.The.sendEvent("Excel.ChartDeactivateEvent " + this.activeChart.Name);

            OfficeConsole.The.sendEvent("Excel.ChartActivateEvent " + this.excel.ActiveChart.Name);

          }
          else
          {
            OfficeConsole.The.sendEvent("Excel.ChartDeactivateEvent " + this.activeChart.Name);
          }
          
          this.activeChart = this.excel.ActiveChart;
        }
      }
    }


    /// <summary>
    /// Load an Excel workbook.
    /// </summary>
    /// <param name="file">the filename of the workbook</param>
    public bool load(string file)
    {
      if (this.excel == null) return false;

      try
      {
        this.workbook = excel.Workbooks.Open(file,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing,
                                          missing);

        this.worksheet = (_Worksheet) this.workbook.ActiveSheet;
        
        //Thread.Sleep(5000);
        return true;

      } 
      catch 
      { 
        this.workbook = null; 
      }

      return false;
    }

    /// <summary>
    /// Create a new worksheet
    /// </summary>
    /// <param name="name">the name of the new worksheet</param>
    public void createWorksheet(string name)
    {
      if (this.excel == null) return;

      _Workbook oldWorkbook = this.workbook;
      _Worksheet oldWorksheet = this.worksheet;

      try
      {
        workbook = excel.Workbooks.Add(XlWBATemplate.xlWBATWorksheet);
        worksheet = (_Worksheet)workbook.ActiveSheet;

        worksheet.Visible = XlSheetVisibility.xlSheetVisible;

        worksheet.Name = name;

        #region Example
        //            Microsoft.Office.Interop.Excel.Range worksheetRange;
        //            string[] cellValue = new string[4];

        //            cellValue[0] = "Company A";
        //            cellValue[1] = "Company B";
        //            cellValue[2] = "Company C";
        //            cellValue[3] = "Company D";
        //            worksheetRange = worksheet.get_Range("A2", "D2");
        //#if OFFICEXP
        //            worksheetRange.set_Value(Missing.Value, cellValue);
        //#else
        //            worksheetRange.Value = cellValue;
        //#endif

        //            double[] dcv = new double[4];
        //            dcv[0] = 75.0;
        //            dcv[1] = 14.0;
        //            dcv[2] = 7.0;
        //            dcv[3] = 4.0;
        //            worksheetRange = worksheet.get_Range("A3", "D3");
        //#if OFFICEXP
        //            worksheetRange.set_Value(Missing.Value, dcv);
        //#else
        //      worksheetRange.Value = dcv;
        //#endif

        //            worksheetRange = worksheet.get_Range("A2:D3", Missing.Value);
        //            Thread.Sleep(2000);  //Let it hang around for a few seconds.
        //            Microsoft.Office.Interop.Excel._Chart chart = (_Chart)workbook.Charts.Add(Missing.Value, Missing.Value, Missing.Value, Missing.Value);
        //            chart.Name = "A quick chart";
        //            chart.ChartWizard(worksheetRange, (int)XlChartType.xl3DPie, 7, (int)XlRowCol.xlRows, 1, 0, 2, "Market Share", Missing.Value, Missing.Value, Missing.Value);

        //            chart.Visible = XlSheetVisibility.xlSheetVisible;

        //            //		Here is the way to call _Workbook.Worksheet's default property by indexer syntax.
        //            Microsoft.Office.Interop.Excel._Worksheet worksheet2 = (_Worksheet)workbook.Worksheets["Market Share!"];
        //            ((_Worksheet)workbook.Worksheets["Market Share!"]).Name = "Fred";
        #endregion
      }
      catch
      {
        this.workbook = oldWorkbook;
        this.worksheet = oldWorksheet;
      }

    }

    public bool runMacro(string macroName)
    {
      if (this.excel == null) return false;
      try
      {
        this.excel.Run(macroName, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing);
        return true;
      } catch(System.Exception) {
        return false;
      }

    }

    /// <summary>
    /// Activates a workbook via its name.
    /// </summary>
    /// <param name="name">The name of the workbook</param>
    /// <returns>true if the workbook was found, false otherways</returns>
    public bool activateWorkbook(string name)
    {
      foreach (_Workbook workbook in excel.Workbooks)
      {
        if (workbook.Name == name)
        {
          workbook.Activate();
          this.workbook = workbook;
          this.worksheet = (_Worksheet) workbook.ActiveSheet;
          return true;
        }
      }
      return false;
    }

    /// <summary>
    /// Activates a worksheet on the current active workbook via its name.
    /// </summary>
    /// <param name="name">The name of the workbook</param>
    /// <returns>true if the worksheet was found, false otherways</returns>
    public bool activateWorksheet(string name)
    {
      foreach (_Worksheet worksheet in excel.ActiveWorkbook.Sheets)
      {
        if (worksheet.Name == name)
        {
          worksheet.Activate();
          this.worksheet = worksheet;
          return true;
        }
      }
      return false;
    }

    public string ActiveWorkbook
    {
      get
      {
        if (this.excel == null)
          return "";
        else
          return this.excel.ActiveWorkbook.Name;
      }
    }

    public string ActiveWorksheet
    {
      get
      {
        if (this.excel == null)
          return "";
        else
          try
          {
            Worksheet sheet = (Worksheet) this.excel.ActiveSheet;
            return sheet.Name;
          }
          catch (System.InvalidCastException)
          {
            return "";
          }

      }
    }

    public string[,] getRange(string from, string to)
    {
      if (this.excel == null || this.worksheet == null) return null;
      try
      {
        Range range = this.worksheet.Cells.get_Range(from, to);
        string[,] rv = new string[range.Rows.Count, range.Columns.Count];

        for (int row = 0; row < range.Rows.Count; ++row)
          for (int column = 0; column < range.Columns.Count; ++column )
          {
            rv[row, column] = ((Range) range.Cells[row + 1, column + 1]).Value2.ToString();
          }
        
        return rv;
      }
      catch (Exception)
      {
        return null;
      }
    }

    public string[,] getRange(string sheet, string from, string to)
    {
      if (this.excel == null || this.workbook == null) return null;
      try
      {
        _Worksheet wsheet = (_Worksheet) this.workbook.Sheets[sheet];
        if (wsheet == null)
        {
          MessageBox.Show("Worksheet not found " + sheet);
          return null;
        }

        Range range = wsheet.Cells.get_Range(from, to);
        string[,] rv = new string[range.Rows.Count, range.Columns.Count];

        for (int row = 0; row < range.Rows.Count; ++row)
          for (int column = 0; column < range.Columns.Count; ++column)
          {
            rv[row, column] = ((Range) range.Cells[row + 1, column + 1]).Value2.ToString();
          }
        
        return rv;
      }
      catch (Exception)
      {
        return null;
      }
    }

    /// <summary>
    /// Quit Excel
    /// </summary>
    public void quit()
    {
      if (excel == null) return;
      try
      {
        _Workbook workbook = excel.ActiveWorkbook;
        if (workbook != null) workbook.Close(false, Missing.Value, false);
      }
      catch { }

      excel.Quit();
      excel = null;
    }

    public bool save(string filename)
    {
      return false;
    }


    // SheetChange event handler
    void cellChanged(object sender, Range r)
    {
      OfficeConsole.The.sendEvent("Excel.CellChangedEvent " + r.Worksheet.Name + "!" + r.Column + ":" + r.Row + " " + r.Value2.ToString());
    }

    // SheetSelectionChange event handler
    void sheetSelectionChange(object sheet, Range r)
    {
      OfficeConsole.The.sendEvent("Excel.SheetSelectionChangeEvent " + r.Worksheet.Name + "!" + r.Column + ":" + r.Row + " " + r.Value2.ToString());
    }

    // SheetActivate event handler
    void sheetActivate(object target)
    {
      try
      {
        Worksheet sheet = (Worksheet) target;
        this.worksheet = sheet;

        foreach (Shape shape in this.worksheet.Shapes)
        {
          if (shape.HasChart == Microsoft.Office.Core.MsoTriState.msoTrue)
          {
          }
        }

        OfficeConsole.The.sendEvent("Excel.SheetActivateEvent " + sheet.Name);
      }
      catch (System.InvalidCastException)
      {
      }
    }

    void afterCalculate()
    {
      OfficeConsole.The.sendEvent("Excel.AfterCalculateEvent");
    }

    void workbookOpen(Workbook Wb)
    {
      OfficeConsole.The.sendEvent("Excel.WorkbookOpenEvent " + Wb.Name);
    }

    void workbookActivate(Workbook Wb)
    {
      OfficeConsole.The.sendEvent("Excel.WorkbookActivateEvent " + Wb.Name);
      this.workbook = Wb;
      this.worksheet = (_Worksheet) Wb.ActiveSheet;
    }

    void workbookDeactivate(Workbook Wb)
    {
      OfficeConsole.The.sendEvent("Excel.WorkbookDeactivateEvent " + Wb.Name);
      if (this.workbook == Wb)
      {
        this.workbook = null;
        this.worksheet = null;
      }
    }

    void windowActivate(Workbook Wb, Window Wn)
    {
      OfficeConsole.The.sendEvent("Excel.WindowActivateEvent " + Wn.Caption + "/" + Wb.Name);
      this.workbook = Wb;
      this.worksheet = (_Worksheet) Wb.ActiveSheet;
    }

    void windowDeactivate(Workbook Wb, Window Wn)
    {
      OfficeConsole.The.sendEvent("Excel.WindowDeactivateEvent " + Wn.Caption + "/" + Wb.Name);
      if (this.workbook == Wb)
      {
        this.workbook = null;
        this.worksheet = null;
      }
    }

    void sheetCalculate(object Sh)
    {
      try
      {
        Worksheet sheet = (Worksheet) Sh;
        OfficeConsole.The.sendEvent("Excel.SheetCalculateEvent " + sheet.Name);
      }
      catch (System.InvalidCastException)
      {
      }

    }


  }
}
