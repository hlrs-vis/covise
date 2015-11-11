#include "tablemodel.h"

#include <QDebug>


/*****
 * C L A S S  ListStrListLessThan
 *****/

class ListStrListLessThan
{
public:
   inline ListStrListLessThan(const int& col) : column(col) {}

   inline bool operator()(const QStringList& row1,
         const QStringList& row2) const
   {
      return QString::localeAwareCompare(row1[column], row2[column]) < 0;
   }

private:
   int column;
};


/*****
 * C L A S S  ListStrListGreaterThan
 *****/

class ListStrListGreaterThan
{
public:
   inline ListStrListGreaterThan(const int& col) : column(col) {}

   inline bool operator()(const QStringList& row1,
         const QStringList& row2) const
   {
      return QString::localeAwareCompare(row1[column], row2[column]) > 0;
   }

private:
   int column;
};


/*****
 * C L A S S  ListStrListTModel
 *****/

/*****
 * constructor
 *****/

ListStrListTModel::ListStrListTModel(QObject* parent) :
   QAbstractTableModel(parent)
{
   //disable sorting before a column is selected
   sortColumn = -1;

   //set variable
   sortOrder = Qt::AscendingOrder;
}


/*****
 * public functions
 *****/

int ListStrListTModel::rowCount(const QModelIndex& /*parent*/) const
{
   return listStrList.size();
}

int ListStrListTModel::columnCount(const QModelIndex& /*parent*/) const
{
   // sollte 3 Spalten sein
//   return 3;
   return tableHeader.size();
}

Qt::ItemFlags ListStrListTModel::flags(const QModelIndex &index) const
{
   Qt::ItemFlags flags = QAbstractTableModel::flags(index);
//   if (index.column() < tableHeader.size())
//   {
//      flags |= Qt::ItemIsEditable;
//   }
   return flags;
}

//Index der Tabellenzellen erzeugen
//
QModelIndex ListStrListTModel::index(int row, int column,
      const QModelIndex& /* parent */) const
{
   if (row < 0 || row > listStrList.size())
   {
      return QModelIndex();
   }

   if (column < 0 || column >= tableHeader.size())
   {
      return QModelIndex();
   }

   return createIndex(row, column);
}

//Anzeigen der Daten pro Zelle[row][column]
//
QVariant ListStrListTModel::data(const QModelIndex& index, int role) const
{
   if (!index.isValid())
   {
      return QVariant();
   }

   if (index.row() >= listStrList.size())
   {
      return QVariant();
   }

   if (role == Qt::DisplayRole)
   {
      return listStrList.at(index.row()).at(index.column());
   }
   else
   {
      return QVariant();
   }
}

//Setzen der Daten geschieht zeilenweise und _nicht_ pro Zelle[row][column]
//
bool ListStrListTModel::setData(const QModelIndex& index,
      const QVariant& value, int role)
{
   if (index.isValid() && role == Qt::DisplayRole)
   {
      if (index.row() >= listStrList.size())
      {
         beginInsertRows(QModelIndex(), index.row(), index.row());
         listStrList.append(value.toStringList());
         endInsertRows();
      }
      else
      {
         listStrList[index.row()] = value.toStringList();
      }

      emit dataChanged(index, index);

      //sort listStrList
      sort(sortColumn, sortOrder);

      return true;
   }

   return false;
}

//Sortieren der listStrList
//
void ListStrListTModel::sort(int column, Qt::SortOrder order)
{
   sortColumn = column;
   sortOrder = order;

   if (sortColumn >= 0)
   {
      ListStrListLessThan lt(sortColumn);
      ListStrListGreaterThan gt(sortColumn);

      if (sortOrder == Qt::AscendingOrder)
      {
         qStableSort(listStrList.begin(), listStrList.end(), lt);
      }
      else
      {
         qStableSort(listStrList.begin(), listStrList.end(), gt);
      }

      if (listStrList.size() > 0)
      {
         QModelIndex topLeft = index(0,0);
         QModelIndex bottomRight =
               index(listStrList.size() - 1, tableHeader.size() - 1);

         emit dataChanged(topLeft, bottomRight);
      }
   }
}

//alle Daten loeschen
//
bool ListStrListTModel::clearData()
{
   if (listStrList.size() > 0)
   {
      return removeRows(0, listStrList.size());
   }
   else
   {
      return false;
   }
}

//Zeile am Ende einfuegen
//
bool ListStrListTModel::appendData(const QStringList& data)
{
   if (data.size() == tableHeader.size())
   {
      int indexAppendRow = listStrList.size();
      QModelIndex appendIndex = index(indexAppendRow, 0);
      setData(appendIndex, data);
      return true;
   }
   else
   {
      return false;
   }
}

bool ListStrListTModel::insertRow(int row,
      const QModelIndex& parent)
{
   QStringList emptyRow;
   for (QStringList::size_type i = 0; i < tableHeader.size(); ++i)
   {
      emptyRow << QString();
   }

   if (row >= 0 && row <= listStrList.size())
   {
      beginInsertRows(parent, row, row);
      listStrList.insert(row, emptyRow);
      endInsertRows();

      return true;
   }

   return false;
}

bool ListStrListTModel::removeRow(int row,
      const QModelIndex& parent)
{
   if (row >= 0 && row < listStrList.size())
   {
      beginRemoveRows(parent, row, row);
      listStrList.removeAt(row);
      endRemoveRows();

      return true;
   }

   return false;
}

bool ListStrListTModel::removeRows(int row, int count,
      const QModelIndex& parent)
{
   if (row < 0  || row >= listStrList.size())
   {
      return false;
   }

   int endRow = row + count -1;
   if (endRow >= listStrList.size())
   {
      endRow = listStrList.size() - 1;
   }

   beginRemoveRows(parent, row, endRow);
   for (int i = 0; i < count; ++i)
   {
      listStrList.removeAt(row);
   }
   endRemoveRows();

   return true;

}

//Anzeigen der HeaderDaten pro Spalte
//
QVariant ListStrListTModel::headerData(int section,
      Qt::Orientation orientation, int role) const
{
   if (role != Qt::DisplayRole)
   {
      return QVariant();
   }

   if (orientation == Qt::Horizontal)
   {
      return tableHeader.at(section);
   }
   else
   {
      return QVariant();
   }
}

//Setzen der HeaderDaten pro Spalte
//
bool ListStrListTModel::setHeaderData(int section, Qt::Orientation orientation,
      const QVariant &value, int role)
{
   if (role != Qt::DisplayRole)
   {
      return false;
   }

   if (section > tableHeader.size())
   {
      return false;
   }

   if (orientation == Qt::Horizontal)
   {
      if (section < tableHeader.size())
      {
         tableHeader[section] = value.toString();
      }
      else
      {
         tableHeader.append(value.toString());
      }

      emit headerDataChanged(orientation, section, section);

      return true;
   }
   else
   {
      return false;
   }
}

QList<QStringList> ListStrListTModel::getListStrList() const
{
   return listStrList;
}
