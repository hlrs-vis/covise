      subroutine norm_in_sparse(norm,zeilen,spalten,sparse,name)

!      use visco_declaration
      use arithmetic
      use declaration

implicit none

      type(sparse_matrix)                    :: sparse
      integer                                :: i,j,k,spalten,zeilen
      real(double),dimension(zeilen,spalten) :: norm
      character(*),optional                  :: name
      real(double),pointer,dimension(:)      :: ele
      integer,dimension(:),pointer           :: row,col
      integer                                :: rowmax,colmax

      k=0
      do i=1,spalten
       do j=1,zeilen
        if (norm(j,i) /= 0) k=k+1
       enddo
      enddo

      allocate(ele(k))
      allocate(row(k))
      allocate(col(k))

      k=0
      rowmax=0
      colmax=0
      do i=1,spalten
       do j=1,zeilen
        if (norm(j,i) /= 0) then
         k=k+1
         ele(k)=norm(j,i)
         row(k)=j
         col(k)=i
         if (j>rowmax) rowmax=j
         if (i>colmax) colmax=i
        endif
       enddo
      enddo

      call matrix_def(sparse,name,ele,row,col,rowmax,colmax)


      deallocate(ele)
      deallocate(row)
      deallocate(col)


      end subroutine norm_in_sparse


subroutine order_jagged(ele,row,col,maxele,dim,dim1,dim2)

use declaration
implicit none
!use visco_declaration
!use arithmetic


integer                            :: maxele
integer,dimension(maxele)               :: col,row
real(double),dimension(maxele)          :: ele
integer                            :: dim,dim1,dim2,i,j,k,alt,alt2
integer,dimension(:,:),pointer,save     :: order
integer,dimension(:),pointer,save       :: col2
real(double),dimension(:),pointer,save  :: ele2


if (dim2.eq.10) then
 allocate(order(dim*5,10))
 allocate(ele2(dim*5*5))
 allocate(col2(dim*5*5))
endif

order=0
ele2=0
col2=0

do i=1,maxele
 j=1
 do
  if (order(row(i),j).eq.0) exit
  if (col(order(row(i),j)).gt.col(i)) exit
  j=j+1
 enddo
 alt2=order(row(i),j)
 k=j
 do
  if (j.eq.dim2) exit
  if (order(row(i),j).eq.0) exit
  alt=alt2
  j=j+1
  alt2=order(row(i),j)
  order(row(i),j)=alt
 enddo
 order(row(i),k)=i
enddo


k=0
do j=1,dim2
 do i=1,dim1
  if (order(i,j).ne.0) then
   k=k+1
   ele2(k)=ele(order(i,j))
   col2(k)=col(order(i,j))
   row(k)=i
  endif
 enddo
enddo

k=0
do i=1,maxele
 if (ele2(i).ne.0) then
  k=k+1
  ele(k)=ele2(i)
  col(k)=col2(i)
  row(k)=row(i)
 endif
enddo
maxele=k

if (dim2.eq.5) then
 deallocate(order)
 deallocate(ele2)
 deallocate(col2)
endif


end subroutine
  
   
