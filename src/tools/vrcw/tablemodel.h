#ifndef TABLEMODEL_H_
#define TABLEMODEL_H_

#include <QAbstractTableModel>
#include <QStringList>

class ListStrListLessThan;
class ListStrListGreaterThan;


class ListStrListTModel: public QAbstractTableModel
{
   Q_OBJECT


public:
   /*****
    * constructor
    *****/
   ListStrListTModel(QObject* parent = 0);


   /*****
    * functions
    *****/
   int rowCount(const QModelIndex& parent) const;
   int columnCount(const QModelIndex& parent) const;

   Qt::ItemFlags flags(const QModelIndex& index) const;
   QModelIndex index(int row, int column,
         const QModelIndex& parent = QModelIndex()) const;
   QVariant data(const QModelIndex& index, int role) const;
   bool setData(const QModelIndex& index, const QVariant& value,
         int role = Qt::DisplayRole);
   void sort(int column, Qt::SortOrder order = Qt::AscendingOrder);
   bool clearData();
   bool appendData(const QStringList& data);

   bool insertRow(int row, const QModelIndex& parent = QModelIndex());
   bool removeRow(int row, const QModelIndex& parent = QModelIndex());
   bool removeRows(int row, int count,
         const QModelIndex & parent = QModelIndex());

   QVariant headerData(int section, Qt::Orientation orientation,
         int role = Qt::DisplayRole) const;
   bool setHeaderData(int section, Qt::Orientation orientation,
         const QVariant& value, int role = Qt::EditRole);

   QList<QStringList> getListStrList() const;


private:
   /*****
    * variables
    *****/
   QList<QStringList> listStrList;
   QStringList tableHeader;
   int sortColumn;
   Qt::SortOrder sortOrder;

};

#endif /* TABLEMODEL_H_ */
