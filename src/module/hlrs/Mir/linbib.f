module declaration2

  use declaration
 
!! This version contains the whole library
!!     date 30.6.93
!! 
!! there are some changes in the assignment routines
!! it is not possible to assign identity vectors and matrices !!!!

!!
!!  Typendeklarationen,       13.10.1992
!!

!! ----------------------------------------------------------------
!! Fuer interne Speicherverwaltung
!! ----------------------------------------------------------------
integer,parameter     :: max_dim       = 10000
integer,parameter     :: max_sum       = 240
integer,parameter     :: max_name      = 30
integer,parameter     :: fringe_length = 10
integer,parameter     :: max_parts     = 8
integer,parameter     :: max_charge    = 4

!! ----------------------------------------------------------------
!! defined for used_flag of objects to know if they were touched
!! ----------------------------------------------------------------
real(double),parameter     :: defined    = 44.
real(double),parameter     :: undefined  = 44.
real(double),parameter     :: alloc      = 22. 
real(double),parameter     :: dealloc    = 21. 



type user_block
  character (LEN=max_sum)   :: summary
  integer                   :: summary_length         ! Laenge des Inhalts
end type user_block

type vector_memory
  integer                       :: dim                   ! Laenge
  integer                       :: link
  real(double),pointer          :: field(:)
end type vector_memory

type memo_admin
 type(vector_memory),pointer    :: container
end type memo_admin

!!type vector_part
!!  type(vector_memory),pointer   :: container
!!  real(double)                  :: scal
!!end type vector_part

type matrix_memory
  integer,pointer               :: ng(:)
  integer,pointer               :: offset(:)
  integer,pointer               :: permutation(:)
  integer,pointer               :: column_number(:)
  real(double),pointer          :: element(:)
  integer,pointer               :: hd(:)           !! Pointer auf Hauptdiagonale
  integer                       :: blocksize       !! beschreibt Anzahl der
                                                   !! Bloecke einer Blockmatrix
  integer                       :: link
end type matrix_memory
  
type scalar
  real(double)                  :: used_flag  !! to control if a scalar
                                              !! is touched
  character (LEN=max_name)      :: name
  type(user_block)              :: block
  real(double)                  :: scal  
  logical                       :: error
end type scalar

type vector
  real(double)                  :: alloc_flag
  real(double)                  :: used_flag   !! to control if a vector
                                               !! is touched
  character (LEN=max_name)      :: name
  type(user_block)              :: block
  integer                       :: charge
  integer                       :: dim
  type(vector_memory)           :: container       
  logical                       :: fleeting 
  logical                       :: error
end type vector

type sparse_matrix
  real(double)                  :: used_flag   !! to control if a matrix
                                               !! is touched
  logical,pointer               :: same        !! Gleichheitspruefung
  character (LEN=max_name)      :: name
  type(user_block)              :: block
  type(matrix_memory),pointer   :: container
  integer                       :: maxrow     ! Dimension
  integer                       :: maxcolumn  ! Dimension
  integer                       :: maxng      ! max Elemente in
                                              ! einer Spalte 
  logical                       :: cont_alloc
  logical                       :: error
  logical                       :: fleeting 
end type sparse_matrix


  !! -------------------------------------------------- 
  !! Fringes sind die Fransen, der Felder rc und
  !! element in Matrix_concentrate
  !! Standardlaenge 10 mit Pointer auf den
  !! naechsten Fringe
  !! -------------------------------------------------- 

  type real_fringe
    real(double),dimension(fringe_length)    :: r_fringe
    type(real_fringe),pointer                :: next
  end type real_fringe

  type int_fringe
    integer,dimension(fringe_length)         :: i_fringe
    type(int_fringe),pointer                 :: next
  end type int_fringe

  !! -------------------------------------------------- 
  !! fringe_points enthalten die anzahl der existierenden
  !! fringes einer zeile und einen Pointer auf das
  !! letzte Listenelement
  !! -------------------------------------------------- 

  type fringe_point
    integer                          :: num_fringes
    type(real_fringe),pointer        :: last_elem
    type(int_fringe),pointer         :: last_rc
  end type fringe_point


end module declaration2
!!*************************************************************
module error_handle

use declaration2

character(LEN=20)          :: error_filename
character(LEN=20)          :: output_filename
integer                    :: error_fileflag=0
integer                    :: output_fileflag=0
type(memo_admin)           :: store(max_dim)      ! Diese Felder regeln
integer,dimension(max_dim) :: store_empty=0       ! interne Verwaltung
                                                  ! des Speichers
real(double)               :: c_start,c_stop

contains

!!************************************************************
subroutine output_file(name)

   character(*),optional     :: name

   output_fileflag=1
   if (present(name)) then
     output_filename=name
   else
     output_filename="LinAlg_out"
   endif

end subroutine output_file

!************************************************************
!**************** E r r o r - H a n d l i n g ***************
!************************************************************

subroutine error_file(name)

   character(*),optional    :: name

   error_fileflag=1

   if(present(name)) then
     error_filename=name 
   else
     error_filename="LinAlg_err"
   endif

end subroutine error_file

!************************************************************

subroutine error (reason, name, block_1, dim11, dim12, &
                                block_2, dim21, dim22)

   character(*),intent(in)          :: reason
   character(*),intent(in),optional :: name
   type(user_block),optional        :: block_1,block_2
   integer,optional                 :: dim11,dim12,dim21,dim22
   integer                          :: err_unit

   if (error_fileflag == 1) then
     err_unit=44
     open(unit=err_unit,         &
          file=error_filename,   &
          status='unknown',      &
          position='append')
   else
     err_unit=6
   endif

   if (present(name)) then
     write(err_unit,*)
     write(err_unit,*)
     write(err_unit,*)' LinAlg: Error in ',name
   else
     write(err_unit,*)' LinAlg: Error'
   endif

   write(err_unit,*)'     because of ',reason

   if (present(block_1)) then
     write(err_unit,*)
     write(err_unit,*)'  summary first operand  : ',&
                         block_1%summary(1:block_1%summary_length)
   endif

   if (present(dim11))&
     write(err_unit,*)'     dimension 1 first operand : ',dim11
   if (present(dim12))&
     write(err_unit,*)'     dimension 2 first operand : ',dim12

   if (present(block_2)) then
     write(err_unit,*)
     write(err_unit,*)'  summary second operand : '&
                        ,block_2%summary(1:block_2%summary_length)
   endif

   if (present(dim21))&
     write(err_unit,*)'     dimension 1 second operand : ',dim21
   if (present(dim22))&
     write(err_unit,*)'     dimension 2 second operand : ',dim22

   if (error_fileflag == 1) close (unit=err_unit, &
                                   status='keep')

end subroutine error
!************************************************************
!************************************************************

end module error_handle
module summary

use error_handle

!! contains summary subroutines

contains

!**************************************************************************

subroutine set_summary (zeichen, block1, block2, block3)

!! calculation in summary / subroutines

   type(user_block),intent(in)   :: block1,block2
   type(user_block),intent(out)  :: block3
   character(*),intent(in)       :: zeichen
   integer                       :: len_char

   len_char=len(zeichen)
   block3%summary_length=min(max_sum,&
          block1%summary_length+block2%summary_length+2+len_char)
   block3%summary(1:block3%summary_length)='('//block1%summary&
          (1:block1%summary_length)&
           //zeichen//block2%summary(1:block2%summary_length)//')'

end subroutine set_summary

!**********************************************************

subroutine norm_summary (zeichen, block1, block2)

   type(user_block),intent(in)   :: block1
   type(user_block),intent(out)  :: block2
   character(*),intent(in)       :: zeichen
   integer                       :: len_char

   len_char=len(zeichen)
   block2%summary_length=min(max_sum,&
          block1%summary_length+2*len_char+2)
   block2%summary='('//zeichen//block1%summary(1:block1%summary_length)//&
                    zeichen//')'

end subroutine norm_summary

!**********************************************************

subroutine func_summary (zeichen, block1, block2)

!! calculation in summary / functions

   type(user_block),intent(in)   :: block1
   type(user_block),intent(out)  :: block2
   character(*),intent(in)       :: zeichen
   integer                       :: len_char

   len_char=len(zeichen)
   block2%summary_length = min(max_sum,block1%summary_length+2+len_char)
   block2%summary(1:block2%summary_length) = zeichen//'('//&
                    block1%summary(1:block1%summary_length)//')'

end subroutine func_summary

!**********************************************************

end module summary
module allocation

use summary

contains

!****************************************************************************
!****************************************************************************

subroutine vector_allocate (vec, vec_laenge)

   type(vector), intent(out)  :: vec
   integer, intent(in)        :: vec_laenge


   if (vec%alloc_flag == alloc) then
     
     if (vec%dim /= vec_laenge) then 
       call vector_deallocate(vec)
       allocate(vec%container%field(vec_laenge))
       vec%container%link = 1
     endif
   else
     allocate(vec%container%field(vec_laenge))
     vec%container%link = 1
     vec%alloc_flag     = alloc
   endif

   vec%used_flag  = defined
   vec%dim        = vec_laenge
   vec%fleeting   =.false.
   vec%error      =.false.


end subroutine vector_allocate

!**********************************************************

subroutine vector_deallocate (vec)

   type(vector)                 :: vec


   if (vec%container%link == 1) then
     deallocate (vec%container%field)
   else 
     vec%container%link = vec%container%link-1
   endif

   vec%used_flag  = undefined
   vec%alloc_flag = dealloc

end subroutine vector_deallocate

!**********************************************************

subroutine matrix_allocate(matrix,nglen,elemlen,rowlen)

   type(sparse_matrix),intent(out) :: matrix
   integer,intent(in)              :: nglen,elemlen,rowlen

   matrix%used_flag=defined

   allocate(matrix%container)
   allocate(matrix%container%ng(nglen))
   allocate(matrix%container%offset(nglen))
   allocate(matrix%container%permutation(rowlen))
   allocate(matrix%container%column_number(elemlen))
   allocate(matrix%container%element(elemlen))
   allocate(matrix%container%hd(rowlen))
!!  wird in matrix_concentrate allokiert

   matrix%error          =.false.
   matrix%cont_alloc     =.true.
   matrix%container%link = 1     !! new

end subroutine matrix_allocate

!**********************************************************

subroutine matrix_deallocate(matrix)

   type(sparse_matrix)               :: matrix

   if (matrix%container%link <= 1) then
     deallocate(matrix%container%ng)
     deallocate(matrix%container%offset)
     deallocate(matrix%container%permutation)
     deallocate(matrix%container%column_number)
     deallocate(matrix%container%element)
     deallocate(matrix%container)
   else
     matrix%container%link = matrix%container%link - 1
   endif

end subroutine matrix_deallocate

!!**********************************************************

end module allocation
module assignments

use allocation
!! contains assignment subroutines

contains

!!************************************************************************
!! in expanded assignments:
!!    first arg. must be specified intent(out) or intent(inout)
!!    second arg. must be specified intent(in)
!!************************************************************************

subroutine vec_to_vec(out,in)

   type(vector),intent(in)     :: in
   type(vector),intent(inout)  :: out

   call vector_deallocate(out)
   
   out%container=in%container

   if (.not.in%fleeting) out%container%link = out%container%link + 1

   out%name =in%name
   out%name =in%block%summary
   out%dim  =in%dim

   out%fleeting    =.false.
   out%error       =.false.
   out%alloc_flag  = alloc
   out%used_flag   = defined  !! out is touched from now on

end subroutine vec_to_vec

!**********************************************************
!**********************************************************

subroutine matrix_to_matrix(out,in)

  type(sparse_matrix),intent(inout) :: out
  type(sparse_matrix),intent(in)    :: in


  if (in%used_flag == defined) then
  
  if ((out%used_flag == defined).and.(out%cont_alloc)) &
       call matrix_deallocate (out)
  
      out%container=>in%container
      out%cont_alloc           =.true.
      out%block%summary        = in%block%summary
      out%block%summary_length = in%block%summary_length
      out%maxrow               = in%maxrow
      out%maxcolumn            = in%maxcolumn
      out%maxng                = in%maxng
      
      if (.not.in%fleeting)&
          out%container%link = out%container%link + 1
  
      out%fleeting =.false.
      out%error=in%error
  
  else
      call error('right hand side not initialized',&
                 'assignment of matrix and matrix ')
      out%error=.true.
  endif

  out%used_flag = defined

end subroutine matrix_to_matrix

!*****************************************************************************

subroutine array_to_vectyp(vec,values)

   type(vector),intent(inout)   :: vec
   real(double),dimension(:),intent(in) :: values

   call vector_allocate(vec,size(values))
   vec%container%field=values
   vec%container%link=1
   vec%error = .false.

   
   vec%used_flag=defined

end subroutine array_to_vectyp

!**********************************************************
end module assignments
module matrix_help

use assignments

contains

!**********************************************************

subroutine sort(feld,dim,perm,jmax)

   integer,intent(in)                 :: dim
   integer,dimension(:),intent(inout) :: feld
   integer,dimension(:),intent(out)   :: perm
   integer,intent(in)                 :: jmax

   integer,pointer        :: index(:,:)
   integer,pointer        :: kk(:)
   integer,dimension(dim) :: ind
   integer,dimension(dim) :: hilf
   integer                :: zz,jj,i,status

   allocate (kk(0:jmax))
   kk=0
   allocate (index(0:jmax,dim))
   index=0

   !! --------------------------------------------------------------------------
   !! dim bezeichnet die Anzahl der Elemente von sort
   !! index(jj,:) enthaelt alle Stellen des Feldes i, an denen das entsprechende
   !!   Element den Wert jj hat
   !! kk(jj) enthaelt die Anzahl der Elemente von index(jj,:)
   !! ind und hilf sind Hilfsfelder zur Umspeicherung
   !! --------------------------------------------------------------------------

   do i=1,dim
     jj=feld(i)
     kk(jj)=kk(jj)+1
     index(jj,kk(jj))=i
   end do

   zz=0

   do jj=jmax,0,-1
     do i=1,kk(jj)
       zz=zz+1
       ind(zz)=index(jj,i)
       hilf(zz)=feld(ind(zz))
     end do
   end do

   do i=1,dim
     perm(ind(i))=i
   end do

   status=0
   do i=1,dim
     if (feld(i)/=hilf(perm(i))) then
       status=1
     end if
     feld(i)=hilf(i)
   end do

   deallocate(kk)
   deallocate(index)

   !! -----------------------------
   !! Kontrollabfragen
   !! -----------------------------

   do i=1,dim-1
     if (feld(i)<feld(i+1)) status=1
     if (count(perm.eq.perm(i))/=1) status=1
   end do
   if (status == 1) then
     stop 'falsch sortiert'
   endif

   return

end subroutine sort

!**********************************************************
subroutine matrix_concentrate(elem,row,col,num_elements,rmax,cmax,matrix)

   type(sparse_matrix)                :: matrix
   real(double),dimension(:)          :: elem
   integer,dimension(:)               :: row,col
   integer                            :: num_elements,rmax,cmax
!!
   type(real_fringe),dimension(rmax)  :: element
   type(int_fringe), dimension(rmax)  :: rc
   type(real_fringe),pointer          :: r_actual
   type(int_fringe),pointer           :: i_actual
   type(fringe_point),dimension(rmax) :: fringes
   integer,dimension(rmax)            :: index,permindex,perminverse
   integer                            :: rowmax, ii,jj,kk, dim_max
   integer                            :: ind, rowind, ll, off
   logical                            :: control


   rowmax=maxval(row)  !! biggest row index
     if (rowmax .gt. rmax) then
       print*,'Fehler in Matrix_concentrate'
       print*,' maxval(row) = ',rowmax
       print*,' rmax real dimension of matrix = ',rmax
       stop
     endif
 
   do ii=1,rmax      !! nullify all fringes as a first step
     nullify(rc(ii)%next)
     rc(ii)%i_fringe=0
     nullify(element(ii)%next)
     element(ii)%r_fringe=0.
     fringes(ii)%num_fringes=1
     nullify(fringes(ii)%last_elem)
     nullify(fringes(ii)%last_rc)
   enddo

   index=0 
   dim_max=num_elements
 
   do ii=1,dim_max           !! Schleife ueber alle Elemente v. row,col,elem
                             !! Vergleich der Elemente in einer fringe-list
     control=.true.
     rowind=row(ii)
                             !! Loop for the first fringe that is not a pointer
     first: do kk=1,min(index(rowind),fringe_length)
       if (col(ii) == rc(rowind)%i_fringe(kk)) then
         element(rowind)%r_fringe(kk)=element(rowind)%r_fringe(kk) + elem(ii)
         num_elements = num_elements - 1
         control=.false.
         exit first
       endif
     enddo first
                             !! Loop for all the other list elements (fringe parts)
                             !! actual is a pointer on one of the list elements
     ind = fringe_length

     second: do jj=2,fringes(rowind)%num_fringes
                             !! if num_fringes is 1 this loop won't be started
 
       if ((jj == fringes(rowind)%num_fringes).AND.&
           ( MOD(index(rowind),fringe_length)/=0)) then
               ind=MOD(index(rowind),fringe_length)
       endif

       if (jj==2) then
         r_actual => element(rowind)%next
         i_actual => rc(rowind)%next
       else
         if (associated(r_actual%next)) r_actual => r_actual%next
         if (associated(i_actual%next)) i_actual => i_actual%next
       endif
 
       third: do kk=1,ind
         if (col(ii) == i_actual%i_fringe(kk)) then
           r_actual%r_fringe(kk)= &
                      r_actual%r_fringe(kk) + elem(ii)
           num_elements = num_elements - 1
           control = .false.
           exit second
         endif
       enddo third
 
     enddo second

     !! -----------------------------------------
     !! neu - belegung !!
     !! falls Element nicht aufaddiert
     !! -----------------------------------------
     if ((control).AND.(col(ii)/=0).AND.(row(ii)/=0)) then
       if  ((MOD(index(rowind),fringe_length) == 0).AND. &
            (index(rowind) /= 0)) then
 
          if (fringes(rowind)%num_fringes /= 1) then
            allocate (r_actual%next)
            allocate (i_actual%next)
            r_actual => r_actual%next
            i_actual => i_actual%next
          else
            nullify(r_actual)
            nullify(i_actual)
            allocate(element(rowind)%next)
            allocate (rc(rowind)%next)
            r_actual => element(rowind)%next
            i_actual => rc(rowind)%next
          endif
 
          nullify (r_actual%next)
          fringes(rowind)%last_elem => r_actual
          nullify (i_actual%next)
          fringes(rowind)%last_rc => i_actual
          fringes(rowind)%num_fringes = fringes(rowind)%num_fringes + 1
 
        endif
 
        index(rowind) = index(rowind) + 1
        if (fringes(rowind)%num_fringes == 1) then
          element(rowind)%r_fringe(index(rowind)) = elem(ii)
          rc(rowind)%i_fringe(index(rowind))=col(ii)
        else
          if (MOD(index(rowind),fringe_length) /= 0) then
            ind = MOD(index(rowind),fringe_length)
          else
            ind = fringe_length
          endif
 
          fringes(rowind)%last_rc%i_fringe(ind) = col(ii)
          fringes(rowind)%last_elem%r_fringe(ind) = elem(ii)
 
        endif
       endif
     enddo

     !! -----------------------------------------------
     !! Matrixbelegung in jagged diagonal form
     !! maxrow,maxcolumn,maxng,permutation
     !! -----------------------------------------------
     permindex        = index
     matrix%maxrow    = rmax
     matrix%maxcolumn = cmax
     matrix%maxng     = maxval(index)

     call matrix_allocate (matrix,matrix%maxng,num_elements,rmax)
     call sort(permindex,rmax,matrix%container%permutation,matrix%maxng)

     matrix%container%offset(1) = 0
     matrix%container%ng(1)     = rmax-(count(index.LT.1))
 
     !! -----------------------------------------------
     !! Belegung von offset und ng
     !! -----------------------------------------------
     do ii=2,(matrix%maxng)
       matrix%container%ng(ii)     = rmax-(count(index.LT.ii))
       matrix%container%offset(ii) = matrix%container%offset(ii-1)+matrix%container%ng(ii-1)
     end do

     do ii=1,rmax
       perminverse(matrix%container%permutation(ii)) = ii
     enddo

     !! ------------------------------------------------
     !! Belegung von column_number und element
     !! ------------------------------------------------

     do ii=1,rmax
       off = 0
       r_actual => element(perminverse(ii))%next
       i_actual => rc(perminverse(ii))%next
       ind      = fringe_length

       do jj=1,fringes(perminverse(ii))%num_fringes

         if ((jj == fringes(perminverse(ii))%num_fringes)) then
           ind = MOD(index(perminverse(ii))-1,fringe_length)+1

           !! --------------------------------------------------------------------------
           !! Folgende Zeile sollte nach Definition von mod unnoetig sein!
           !! --------------------------------------------------------------------------
           if(index(perminverse(ii)) == 0) ind=0
         endif
         if (jj == 1) then
 
           do ll=1,ind
             off=off+1
             matrix%container%column_number(ii + matrix%container%offset(off))= &
                            rc(perminverse(ii))%i_fringe(ll)
             matrix%container%element(ii + matrix%container%offset(off))= &
                           element(perminverse(ii))%r_fringe(ll)
           enddo
         else
 
         do ll=1,ind
           off=off+1
           matrix%container%column_number(ii + matrix%container%offset(off))= &
                          i_actual%i_fringe(ll)
           matrix%container%element(ii + matrix%container%offset(off))= &
                          r_actual%r_fringe(ll)
         enddo
         if (associated(i_actual%next)) i_actual => i_actual%next
         if (associated(r_actual%next)) r_actual => r_actual%next
 
       endif
     enddo
   enddo

   matrix%used_flag = defined
   call set_hd(matrix)
 
 !! -------------------------------------------------------------------
 !! Schoenheitsfehler: Elemente, die beim Aufaddieren den Wert Null
 !! ergeben, werden als vollwertige Elemente angesehen
 !! -------------------------------------------------------------------
 
end subroutine matrix_concentrate

!**************************************************************************

subroutine set_hd(matrix)


   !! ------------------------------------------------
   !! sets main diagonale for matrix
   !! main diagonale without permutation!!
   !! ------------------------------------------------

   type(sparse_matrix), intent(inout)  :: matrix
   integer                             :: i,j,ii
   integer,dimension(matrix%maxrow)    :: perm_inv  !! inverse permutation

   do i=1,matrix%maxrow
     perm_inv(matrix%container%permutation(i))=i
   enddo

   matrix%container%hd=0

   do i=1,matrix%maxng
     do j=1,matrix%container%ng(i)
       ii = j + matrix%container%offset(i)

       !! --------------------------------------
       !! main diagonale without permutation!!
       !! --------------------------------------

       if (matrix%container%column_number(ii)==perm_inv(j)) &
           matrix%container%hd(perm_inv(j)) = ii

     enddo
   enddo

end subroutine set_hd

!**************************************************************************

end module matrix_help
module definition

use matrix_help

contains

!**********************************************************

subroutine vector_def2(vec,name,values)

   type(vector),intent(out)          :: vec
   character(*),intent(in)           :: name
   real(double),dimension(:),pointer :: values


   call vector_allocate(vec, size(values))

   vec%fleeting             =.false.
   vec%error                =.false.
   vec%block%summary        = " "
   vec%block%summary_length = 1

   vec%container%field=>values

   call mblock_def(vec%block,name)

   vec%name=name

   vec%used_flag      = defined

end subroutine vector_def2

!**********************************************************

subroutine block_def(block,skalar)

   type(user_block),intent(out) :: block
   real(double),intent(in)      :: skalar
   character(8)                 :: string

   write(string,'(E8.2)') skalar
   block%summary=string
   block%summary_length=8

end subroutine block_def

!**********************************************************

subroutine mblock_def(block,name)

   type(user_block),intent(out) :: block
   character(*),intent(in)      :: name
   integer                      :: dim

   dim=LEN(name)
   block%summary(1:dim)=name
   block%summary_length=dim

end subroutine mblock_def

!**********************************************************

subroutine matrix_def(matrix,name,elem,rr,cc,rmax,cmax)

   type(sparse_matrix),intent(out)         :: matrix
   character(*),intent(in)                 :: name
   real(double),dimension(:),intent(in)    :: elem
   integer,dimension(:),intent(in)         :: rr,cc
   integer,intent(in)                      :: rmax,cmax
   integer                                 :: num_elements

   matrix%used_flag=defined

   if ((rmax==0).OR.(cmax==0)) then
     call matrix_allocate(matrix,0,0,rmax)
     call mblock_def(matrix%block,name)
     matrix%name=name
     matrix%maxng=0
     matrix%maxrow=rmax
     matrix%maxcolumn=cmax
     matrix%fleeting =.false.
     return
   end if

   allocate(matrix%container)
   call mblock_def(matrix%block,name)
   matrix%name=name

   num_elements=size(elem)

   call matrix_concentrate(elem,rr,cc,num_elements,rmax,cmax,matrix)

   matrix%fleeting =.false. 
   matrix%cont_alloc=.true.
   matrix%error=.false.

end subroutine matrix_def

!**********************************************************

subroutine matrix_def_inverse(matrix,name,aa,row_num,col_num,rmax,cmax)

  type(sparse_matrix),intent(in)     :: matrix
  character(*)                       :: name
  real(double),dimension(:)          :: aa
  integer,dimension(:)               :: row_num,col_num
  integer,intent(out)                :: rmax,cmax
  integer                            :: i,j,k
  integer                            :: num_elements
  integer,dimension(matrix%maxrow)   :: perm_inv


  if (matrix%used_flag==defined) then

    num_elements=size(matrix%container%element)
    print*,size(aa)

    do i=1,matrix%maxrow
      perm_inv(matrix%container%permutation(i))=i
    end do

    i=1
    do k=1,matrix%maxng
      do j=1,matrix%container%ng(k)
        row_num(i)=perm_inv(j)
        i=i+1
      end do
    end do

    do i=1,num_elements
      aa(i)=matrix%container%element(i)
      col_num(i)=matrix%container%column_number(i)
    end do

    cmax=matrix%maxcolumn
    rmax=matrix%maxrow

    name(1:len(name)) = matrix%block%summary(1:len(name))

  else
    call error('matrix not initialized',&
               'matrix to coordinate format')
  endif

end subroutine matrix_def_inverse

!**********************************************************


end module definition
module matrix

use definition

!! contains operations between matrices

contains 

!************************************************************

function matrix_mult_matrix(matrix_1,matrix_2) &
     result(matrix_3)

   type(sparse_matrix),intent(in)          :: matrix_1,matrix_2
   type(sparse_matrix),pointer             :: matrix_3
   integer,dimension(matrix_1%maxrow)      :: ipa
   integer                                 :: ii,ka,ja,kb,num_elements,dim
   integer,allocatable,dimension(:)        :: row,column
   real(double),allocatable,dimension(:)   :: elem

   allocate(matrix_3)
   matrix_3%error=.false.
   matrix_3%used_flag=defined

   if ((matrix_1%used_flag == defined).and.(matrix_2%used_flag == defined)) then

     if ((matrix_1%error).or.(matrix_2%error)) then
       matrix_3%error=.true.
       return
     endif

     if (matrix_1%maxcolumn/=matrix_2%maxrow) then
       call error('incompatible matrices','multiplikation',&
                   matrix_1%block,matrix_1%maxrow,matrix_1%maxcolumn,&
                   matrix_2%block,matrix_2%maxrow,matrix_2%maxcolumn)
       matrix_3%error=.true.
       return
     end if

     matrix_3%name='matrix_mult_matrix'

     if ((matrix_1%maxrow==0).OR.(matrix_2%maxrow==0).OR.(matrix_2%maxcolumn==0)) then
       call matrix_allocate(matrix_3,0,0,matrix_1%maxrow)
       call set_summary('*',matrix_1%block,matrix_2%block,matrix_3%block)
       matrix_3%maxng=0
       matrix_3%maxrow=matrix_1%maxrow
       matrix_3%maxcolumn=matrix_2%maxcolumn
       matrix_3%fleeting =.true.
       return
     end if

     dim=matrix_1%maxng * matrix_1%container%ng(1) * matrix_2%maxng

     allocate(row(dim))
     allocate(column(dim))
     allocate(elem(dim))
     row=0
     column=0
     elem=0

     do ii=1,matrix_1%maxrow
       ipa(matrix_1%container%permutation(ii))=ii
     end do

     ii=0
     do ka=1,matrix_1%maxng
       do kb=1,matrix_2%maxng
         do ja=1,matrix_1%container%ng(ka)
           if ((matrix_2%container%permutation&
               (matrix_1%container%column_number(matrix_1%container%offset(ka)+ja)))&
                <=matrix_2%container%ng(kb)) then

             ii=ii+1

             elem(ii)=matrix_1%container%element(matrix_1%container%&
                      offset(ka)+ja)&
                    * matrix_2%container%element(matrix_2%container%offset(kb)+&
                      matrix_2%container%permutation(matrix_1%container%  &
                      column_number(matrix_1%container%offset(ka)+ja)))
             row(ii)=ipa(ja)
             column(ii)=matrix_2%container%column_number(matrix_2%container%offset&
                       (kb)+ matrix_2%container%permutation &
                       (matrix_1%container%column_number(matrix_1%&
                       container%offset(ka)+ja)))
           end if
         end do
       end do
     end do

     num_elements=ii

     call matrix_concentrate(elem,row,column,num_elements,matrix_1%maxrow,&
                             matrix_2%maxcolumn,matrix_3)
     call set_summary('*',matrix_1%block,matrix_2%block,matrix_3%block)
     matrix_3%fleeting =.true.

     deallocate(row)
     deallocate(column)
     deallocate(elem)

   else
     call error('at least one object not initialized',&
                'matrix multiplication' )
     matrix_3%error=.true.
   endif

end function matrix_mult_matrix

!**********************************************************

subroutine dump_matrix(matrix)

   type(sparse_matrix),intent(in)  :: matrix
   integer                         :: out_unit

   if (output_fileflag == 1) then
     out_unit=45
     open(unit=out_unit,          &
          file=output_filename,   &
          status='unknown',       &
          position='append')
   else
     out_unit=6
   endif

   write(out_unit,*)'matrix%maxrow=',matrix%maxrow
   write(out_unit,*)'matrix%maxcolumn=',matrix%maxcolumn
   write(out_unit,*)'matrix%maxng=',matrix%maxng
   write(out_unit,*)'matrix%container%offset=',matrix%container%offset
   write(out_unit,*)'matrix%container%ng=',matrix%container%ng
   write(out_unit,*)'matrix%container%permutation=',matrix%container%permutation
   write(out_unit,*)'matrix%container%column_number=',matrix%container%column_number
   write(out_unit,*)'matrix%container%element=',matrix%container%element
   write(out_unit,*)'matrix%container%hd=',matrix%container%hd
   write(out_unit,*)'matrix%used_flag = ',matrix%used_flag 
   write(out_unit,*)'matrix%cont_alloc = ',matrix%cont_alloc 
   write(out_unit,*)'matrix%error = ',matrix%error 
   write(out_unit,*)'matrix%fleeting = ',matrix%fleeting

   if (output_fileflag == 1) then
     close(unit=out_unit)
   endif

end subroutine dump_matrix

!**********************************************************

subroutine matrix_summary(matrix)

type(sparse_matrix),intent(in) :: matrix
integer                        :: out_unit

if (output_fileflag == 1) then
   out_unit=45
   open(unit=out_unit,         &
        file=output_filename,   &
        status='unknown',        &
        position='append')
else
   out_unit=6
endif

write(out_unit,*)
write(out_unit,*)' Name  :',matrix%name
write(out_unit,*)' Inhalt :',matrix%block%summary(1:matrix%block%summary_length)
write(out_unit,*)

if (output_fileflag == 1) close (unit=out_unit,&
                                 status='keep')

end subroutine matrix_summary

!**********************************************************

subroutine print_matrix(matrix)

   type(sparse_matrix),intent(in)                    :: matrix
   real(double),dimension(:),pointer                 :: elem
   integer,dimension(:),pointer                      :: row, col
   integer                                           :: rmax, cmax, imax
   integer                                           :: out_unit
   character*20                                      :: name

   if (matrix%used_flag==defined) then

     if (output_fileflag == 1) then
       out_unit=45
       open(unit=out_unit,     &
       file=output_filename,   &
       status='unknown',       &
       position='append')
     else
       out_unit=6
     endif

     imax = size(matrix%container%element)
     allocate(row(imax),col(imax),elem(imax))
     call matrix_def_inverse(matrix,name,elem(1:imax),row(1:imax),col(1:imax),&
                                              rmax,cmax)
     write(out_unit,*)
     write(out_unit,*)' matrix ',matrix%name
     write(out_unit,*)' element = ',elem
     write(out_unit,*)' row_num = ',row
     write(out_unit,*)' col_num = ',col
     write(out_unit,*)

   else
     call error('matrix not initialized',&
                'print_matrix')
   endif

end subroutine print_matrix

!**********************************************************

subroutine print_full_matrix(matrix)

   type(sparse_matrix),intent(in)                            :: matrix 
   real(double),dimension(size(matrix%container%element))    :: elem
   real(double),dimension(:,:),allocatable                   :: AA
   integer,dimension(size(matrix%container%element))         :: row,col
   integer                                                   :: rmax,cmax,num_elements
   integer                                                   :: out_unit
   character*20                                              :: name

   if (matrix%used_flag==defined) then

     if (output_fileflag == 1) then
       out_unit=45
       open(unit=out_unit,     &
       file=output_filename,   &
       status='unknown',       &
       position='append')
     else
       out_unit=6
     endif
    
     call matrix_def_inverse(matrix,name,elem,row,col,rmax,cmax)
     allocate (AA(rmax,cmax))
     AA=0.
     num_elements=size(elem)

     do i=1,num_elements
       AA(row(i),col(i))=elem(i)
     enddo
    
     write(out_unit,*)
     write(out_unit,*)'matrix ',matrix%name
     do i=1,rmax
       write(out_unit,*) AA(i,:)
     enddo
     write(out_unit,*)
    
   else
     call error('matrix not initialized',&
                'print_full_matrix')
   endif

end subroutine print_full_matrix

!**********************************************************
!******************************************************************************

end module matrix
module vektor

use matrix

!! contains subroutines for vector & vector and vector & skalar

contains
!**********************************************************

function vec_minus_vec(vec_1,vec_2)&
         result(vec_3)

   type(vector),intent(in) :: vec_1,vec_2
   type(vector)            :: vec_3
   integer                 :: i,k


   vec_3%dim       = vec_1%dim
   vec_3%error     =.false.
   vec_3%used_flag = defined

   if ((vec_1%used_flag == defined).and.(vec_2%used_flag == defined)) then

     if ((vec_1%error).or.(vec_2%error)) then
       vec_3%error=.true.
       return
     endif

     if (vec_1%dim/=vec_2%dim) then
       call error('incompatible dimensions','subtraction',&
                   vec_1%block,vec_1%dim,&
                   block_2=vec_2%block,dim21=vec_2%dim)
       vec_3%error=.true.
       return
     endif

     vec_3%name='vec_minus_vec'

     call set_summary('-',vec_1%block,vec_2%block,vec_3%block)

     !! ----------------------------------------------------------------
     !! Alle Adressen der Eingangsvektoren werden in vec_3 eingetragen
     !! und die links auf die Felder erhoeht.
     !! ----------------------------------------------------------------


     call vector_allocate(vec_3,vec_3%dim)

     vec_3%container%field  = vec_1%container%field - vec_2%container%field

     !! ----------------------------------------------------------------
     !! Wenn der neue Vektor mehr als halbvoll ist: zusammenaddieren
     !! ----------------------------------------------------------------

     vec_3%fleeting =.TRUE.

     if (vec_1%fleeting ) call vector_deallocate(vec_1)
     if (vec_2%fleeting ) call vector_deallocate(vec_2)

   else
     call error('at least one vector not initialized','substraction')
     vec_3%error=.true.
   endif

end function vec_minus_vec

!**********************************************************
!**********************************************************

subroutine dump_vec(vec)

   type(vector),intent(in) :: vec
   integer                 :: i

   print*,' vector ',vec%name
   call vector_summary(vec)
   print*,'      vec%usedflag  = ',vec%used_flag
   print*,'      vec%fleeting  = ',vec%fleeting
   print*,'      vec%error     = ',vec%error

    call print_vector(vec)

end subroutine dump_vec

!**********************************************************

subroutine vector_summary(vec)

   type(vector),intent(in) :: vec
   integer                 :: out_unit

   if (output_fileflag == 1) then
      out_unit=45
      open(unit     = out_unit,          &
           file     = output_filename,   &
           status   ='unknown',          &
           position ='append')
   else
      out_unit=6
   endif

   write(out_unit,*)
   write(out_unit,*)' Name  :',vec%name
   write(out_unit,*)' Inhalt :',vec%block%summary(1:vec%block%summary_length)
   write(out_unit,*)

   if (output_fileflag == 1) close (unit=out_unit,&
                                    status='keep')

end subroutine vector_summary

!**********************************************************

subroutine print_vector(vec)
 
   type(vector),intent(in) :: vec
   integer                 :: i,out_unit

   if (vec%used_flag==defined) then

     if (output_fileflag == 1) then
       out_unit=45
       open(unit     = out_unit,          &
            file     = output_filename,   &
            status   ='unknown',          &
            position ='append')
     else
       out_unit=6
     endif

     write(out_unit,*)
     write(out_unit,*)' vector ',vec%name

     write(out_unit,*)(vec%block%summary)

     do i=1,vec%dim
       write(out_unit,*)(vec%container%field(i))
     end do
     write(out_unit,*)
 
     if (output_fileflag == 1) close (unit=out_unit,&
                                      status='keep')

   else
     call error('vector not initialized',&
                'print_vector')
   endif

end subroutine print_vector

!**********************************************************


end module vektor
module matvec

use vektor

!! contains subroutines for operations between matrices & vectors

contains

!***************************************************************

function matrix_mult_vec(matrix,vec_1)&
     result(vec_2)

!! not anymore !! in this function a dummy vector is used for the intern addition
!! not anymore !! vec_1 stays as it is with probably more than one container

   type(sparse_matrix),intent(in)        :: matrix
   type(vector),intent(in)               :: vec_1
   type(vector),pointer                  :: vec_2
   real(double),dimension(matrix%maxrow) :: yy
   integer                               :: i,ii,k

   allocate(vec_2)
   vec_2%dim       = vec_1%dim
   vec_2%error     =.false.
   vec_2%used_flag = defined

   if ((vec_1%used_flag == defined).and.(matrix%used_flag == defined)) then

      if ((vec_1%error).or.(matrix%error)) then
         vec_2%error=.true.
         return
      endif
  
      if (matrix%maxcolumn /= vec_1%dim) then
         call error('incompatible dimensions','multiplikation',&
                     matrix%block,matrix%maxrow,matrix%maxcolumn,&
                     vec_1%block,vec_1%dim)
         vec_2%error=.true.
         return
      endif
  
      vec_2%name='matrix_mult_vec'
  
      call vector_allocate(vec_2,matrix%maxrow)

      !! ---------------------------------------
      !! Hier Abfrage fuer Matrix-Dimension 0
      !! ---------------------------------------
      if ((matrix%maxrow /= 0).AND.(matrix%maxcolumn == 0)) then
          vec_2%container%field(1:vec_2%dim)=0.
          call set_summary('*',matrix%block,vec_1%block,vec_2%block)
          vec_2%fleeting =.true.
          return
      end if
  
      yy=0.
      do k=1,matrix%maxng
        do i=1,matrix%container%ng(k)
          ii=i+matrix%container%offset(k)
          yy(i)=yy(i)+matrix%container%element(ii)*&
                vec_1%container%field(matrix%container%column_number(ii))    !! added 9.9.93
        end do
      end do
  
  
      !! -----------------------------------------------------------------
      !! Die einzelnen Elemente werden mit dem Skalar des Vektors vec_1
      !! multipliziert
      !! -----------------------------------------------------------------
  
      do i=1,vec_2%dim
        vec_2%container%field(i) = yy(matrix%container%permutation(i))
      end do
  
      call set_summary('*',matrix%block,vec_1%block,vec_2%block)
  
      if (matrix%fleeting) call matrix_deallocate(matrix)
      if (vec_1%fleeting ) call vector_deallocate(vec_1)

   else
     call error('at least one object not initialized','multiplication &
                 &matrix vector')
     vec_2%error=.true.
   endif

end function matrix_mult_vec

!**********************************************************

end module matvec
module arithmetic

!!
!! Interfaces,      19.08.1992
!!

use matvec
 
interface operator(-)
   module procedure  vec_minus_vec
end interface


interface operator(*)
   module procedure matrix_mult_vec,       &
                    matrix_mult_matrix
end interface


interface assignment(=)
   module procedure vec_to_vec,        &
                    array_to_vectyp,   &
                    matrix_to_matrix
end interface

end module arithmetic










