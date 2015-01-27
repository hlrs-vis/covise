
function matrix_mult_vec(matrix,vec_1)&
     result(vec_2)

!! not anymore !! in this function a dummy vector is used for the intern addition
!! not anymore !! vec_1 stays as it is with probably more than one container

type(sparse_matrix),intent(in):: matrix
type(vector),intent(in)       :: vec_1
!!type(vector)                  :: dummy_vec                      !! added 9.9.93
!!type(sparse_matrix)           :: dummy_mat                      !! added 9.9.93
type(vector),pointer          :: vec_2
real(double),dimension(matrix%maxrow) :: yy
integer                       :: i,ii,k

allocate(vec_2)
vec_2%dim=vec_1%dim
vec_2%error=.false.
vec_2%used_flag=defined

if ((vec_1%used_flag == defined).and.(matrix%used_flag == defined)) then

!! not allowed                                                  !! added 9.9.93
 call intern_add(vec_1)                                                  !! added 9.9.93
!! use instead with same effect                                 !! added 9.9.93 
!!   dummy_vec%used_flag=defined                                                !! added 9.9.93
!!   dummy_vec%charge=vec_1%charge                                               !! added 9.9.93
!!   do i=1,vec_1%charge                                                         !! added 9.9.93
!!     dummy_vec%c_b(i)%container=>vec_1%c_b(i)%container                        !! added 9.9.93
!!     dummy_vec%c_b(i)%scal=vec_1%c_b(i)%scal                                   !! added 9.9.93
   !! might have to set link up (depends on internadd,if link is set down there)!! added 9.9.93
   !! dummy_vec%c_b(i)%container%link = dummy_vec%c_b(i)%container%link + 1    !! added 9.9.93
!!   enddo                                                                       !! added 9.9.93
   !! link stays as it is (or not), need scalar for dummy vector this time

!!   call intern_add(dummy_vec)    !! links should be set down there             !! added 9.9.93


  if ((vec_1%error).or.(matrix%error)) then
    vec_2%error=.true.
    return
  endif
  
  if (matrix%maxcolumn/=vec_1%dim) then
    call error('incompatible dimensions','multiplikation',&
                matrix%block,matrix%maxrow,matrix%maxcolumn,&
                vec_1%block,vec_1%dim)
    vec_2%error=.true.
    return
  endif
  
  vec_2%name='matrix_mult_vec'
  
  call vector_allocate(vec_2,matrix%maxrow)
  !! Hier Abfrage fuer Matrix-Dimension 0
  if ((matrix%maxrow/=0).AND.(matrix%maxcolumn==0)) then
    vec_2%c_b(1)%container%field(1:vec_2%dim)=0.
    call set_summary('*',matrix%block,vec_1%block,vec_2%block)
    vec_2%fleeting =.true.
    return
  end if
  
  yy=0.
  do k=1,matrix%maxng
    do i=1,matrix%container%ng(k)
      ii=i+matrix%container%offset(k)
      yy(i)=yy(i)+matrix%container%element(ii)*&
            vec_1%c_b(1)%container%field(matrix%container%column_number(ii))    !! added 9.9.93
!!            dummy_vec%c_b(1)%container%field(matrix%container%column_number(ii))
    end do
  end do
  
  
  !! Die einzelnen Elemente werden mit dem Skalar des Vektors vec_1
  !! multipliziert
  
  do i=1,vec_2%dim
    vec_2%c_b(1)%container%field(i)=yy(matrix%container%permutation(i))* &
                                   vec_1%c_b(1)%scal                            !! added 9.9.93
!!                                   dummy_vec%c_b(1)%scal
  end do
  
  vec_2%c_b(1)%scal=1.0
  vec_2%charge=1
  
  call set_summary('*',matrix%block,vec_1%block,vec_2%block)
  
!!   call vector_deallocate(dummy_vec)                                     !! added 9.9.93
!!   if (matrix%fleeting) then                                         !! added 9.9.93
!!      dummy_mat%used_flag=defined                                    !! added 9.9.93
!!      dummy_mat%container=>matrix%container                              !! added 9.9.93
!!      call matrix_deallocate(dummy_mat)                                   !! added 9.9.93
!!   endif                                                              !! added 9.9.93
!!   if (vec_1%fleeting ) then                                         !! added 9.9.93
!!      dummy_vec%charge=vec_1%charge                                   !! added 9.9.93
!!      do i=1,vec_1%charge                                             !! added 9.9.93
!!        dummy_vec%c_b(i)%container=>vec_1%c_b(i)%container            !! added 9.9.93
!!      enddo
!!      !! link stays as it is, don't need scalar for dummy vector     !! added 9.9.93
!!      call vector_deallocate(dummy_vec)                                  !! added 9.9.93
!!   endif

  if (matrix%fleeting) call matrix_deallocate(matrix)
  if (vec_1%fleeting ) call vector_deallocate(vec_1)

else
  call error('at least one object not initialized','multiplication &
              &matrix vector')
  vec_2%error=.true.
endif

!!print*,' maxtvec : ende'
!!call print_vector(vec_2)
!!call dump_vec(vec_2)

end function matrix_mult_vec
