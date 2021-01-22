module fftw3
  use, intrinsic :: iso_c_binding
  include 'fftw3.f03'
end module fftw3

subroutine init_H(num_pw,H_kinetic,H_local)
  !-------------------------------------------------!
  ! This subroutine sets up the H for the eigenvalue!
  ! problem.                                        !
  !-------------------------------------------------!
  implicit none

  ! Define what we mean by "double precision"
  integer,       parameter                               :: dp=selected_real_kind(15,300)

  integer,                                   intent(in)  :: num_pw
  real(kind=dp), dimension(num_pw),          intent(out) :: H_kinetic
  real(kind=dp), dimension(num_pw),          intent(out) :: H_local

  integer                                                :: np,status

  ! Set the kinetic energy operator in reciprocal-space
  ! NB np=1 is the G=0 element
  !    np=2 has its conjugate at np=num_pw
  !    np=3 has its conjugate at np=num_pw-1
  !                  etc.
  H_kinetic(1) = 0.0_dp
  do np=1,num_pw/2
     H_kinetic(np+1) = 0.5_dp*real(np,kind=dp)**2
     H_kinetic(num_pw-(np-1)) = H_kinetic(np+1)
  end do

  ! Set the local potential operator in real-space (centred on the middle of the 1D cell)
  H_local = 0.0_dp
  do np=1,num_pw
!     ! Gaussian -- good, but very few plane-waves required
!     H_local(np) = -6.0_dp*exp(-((real(np-1,kind=dp)/real(num_pw,kind=dp)-0.5_dp)**2)/(2*0.1_dp**2))

!     ! -1/r -- too many plane-waves required
!     H_local(np) = -1.00_dp/(abs((real(np-1,kind=dp)/real(num_pw,kind=dp)-0.5_dp)))

     ! Sort-of pseudopotential -- ~1/r asymptotically, but finite at nucleus
     H_local(np) = -0.37_dp/(0.005_dp+abs((real(np-1,kind=dp)/real(num_pw,kind=dp)-0.5_dp)))

!    ! Hooke's atom
!     H_local(np) = 10.0_dp*(-1.0_dp+4*((real(np-1,kind=dp)/real(num_pw,kind=dp)-0.5_dp))**2)
  end do

  return

end subroutine init_H

subroutine apply_H(num_pw,num_states,state,H_kinetic,H_local,H_state)
  !-------------------------------------------------!
  ! This subroutine applies the H hermitian_operator!
  !  H to a state.                                  !
  !-------------------------------------------------!
  use fftw3

  implicit none

  ! Define what we mean by "double precision"
  integer, parameter :: dp=selected_real_kind(15,300) ! or could use real64
  
  integer,                                        intent(in)  :: num_pw
  integer,                                        intent(in)  :: num_states
  complex(kind=dp), dimension(num_pw,num_states), intent(in)  :: state
  real(kind=dp),    dimension(num_pw),            intent(in)  :: H_kinetic
  real(kind=dp),    dimension(num_pw),            intent(in)  :: H_local
  complex(kind=dp), dimension(num_pw,num_states), intent(out) :: H_state

  integer                                     :: np
  integer                                     :: nb
  integer                                     :: status
  complex(kind=dp), dimension(:), allocatable :: tmp_state
  complex(kind=dp), dimension(:), allocatable :: tmp_state_in

  type(C_PTR) :: plan                       ! force C-pointer (prob. 64-bit)

  allocate(tmp_state(num_pw),stat=status)
  if(status/=0) stop 'Error allocating tmp_state in apply_H'

  allocate(tmp_state_in(num_pw),stat=status)
  if(status/=0) stop 'Error allocating tmp_state in apply_H'

  ! Loop over the states
  do nb=1,num_states

    ! Compute the contributions from the local potential
    !
    ! First create a plan. We want a forward transform and we want to estimate
    ! (rather than measure) the most efficient way to perform the transform.
!    plan = fftw_plan_dft_1d(num_pw,state(:,nb),tmp_state,FFTW_FORWARD,FFTW_ESTIMATE)
    plan = fftw_plan_dft_1d(num_pw,tmp_state_in,tmp_state,FFTW_FORWARD,FFTW_ESTIMATE)

    ! Compute the FFT of in using the current plan, store in out.
    tmp_state_in = state(:,nb)
    call fftw_execute_dft(plan,tmp_state_in,tmp_state)

    ! Apply the local potential in real-space, remembering normalisation of 1/grid-points
    do np=1,num_pw
       tmp_state(np) = H_local(np)*tmp_state(np)/real(num_pw,kind=dp)
    end do

    ! Now transform back
    !
    ! Create a plan. We want a backward transform and we want to estimate
    ! (rather than measure) most efficient way to perform the transform.
    plan = fftw_plan_dft_1d(num_pw,tmp_state_in,tmp_state,FFTW_BACKWARD,FFTW_ESTIMATE)
!    plan = fftw_plan_dft_1d(num_pw,state(:,nb),tmp_state,FFTW_BACKWARD,FFTW_ESTIMATE)
!    call fftw_plan_dft_1d(plan,num_pw,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! Compute the FFT of in using the current plan, store in out.
    call fftw_execute_dft(plan,tmp_state,H_state(:,nb))

    ! Add in the contribution from the kinetic term
    do np=1,num_pw
       H_state(np,nb) = H_state(np,nb) + H_kinetic(np)*state(np,nb)
    end do

  end do

  deallocate(tmp_state,stat=status)
  if(status/=0) stop 'Error deallocating tmp_state in apply_H'

  return

end subroutine apply_H

subroutine construct_full_H(num_pw,H_kinetic,H_local,full_H)
  !---------------------------------------------------------------!
  ! This subroutine constructs the Hamiltonian hermitian operator !
  ! in full matrix form (in reciprocal-space).                    !
  !---------------------------------------------------------------!
  use fftw3

  implicit none

  ! Define what we mean by "double precision"
  integer, parameter :: dp=selected_real_kind(15,300)

  integer,                                     intent(in)  :: num_pw
  real(kind=dp),    dimension(num_pw),         intent(in)  :: H_kinetic
  real(kind=dp),    dimension(num_pw),         intent(in)  :: H_local
  complex(kind=dp), dimension(num_pw,num_pw),  intent(out) :: full_H

  complex(kind=dp), dimension(:), allocatable :: tmp_state1,tmp_state2
  integer                                     :: np1,np2,status

  type(C_PTR) :: plan                       ! needs to be a 64-bit C pointer

  ! Initialise the output matrix
  full_H = cmplx(0.0_dp,0.0_dp,dp)

  ! First add the contribution from the local potential term

  allocate(tmp_state1(num_pw),stat=status)
  if(status/=0) stop 'Error allocating tmp_stat1 in construct_full_H'

  allocate(tmp_state2(num_pw),stat=status)
  if(status/=0) stop 'Error allocating tmp_stat1 in construct_full_H'

  tmp_state1 = cmplx(0.0_dp,0.0_dp,dp)
  tmp_state2 = cmplx(0.0_dp,0.0_dp,dp)

  do np1=1,num_pw
    ! First create a plan. We want a forward transform and we want to estimate
    ! (rather than measure) the most efficient way to perform the transform.
    plan = fftw_plan_dft_1d(num_pw,tmp_state1,tmp_state2,FFTW_FORWARD,FFTW_ESTIMATE)

    tmp_state1 = cmplx(0.0_dp,0.0_dp,dp)

    tmp_state1(np1) = cmplx(1.0_dp,0.0_dp,dp)

    ! Compute the FFT of in using the current plan, store in out.
    call fftw_execute_dft(plan,tmp_state1,tmp_state2)

    ! Apply the local potential in real-space, remembering normalisation of 1/grid-points
    do np2=1,num_pw
       tmp_state2(np2) = H_local(np2)*tmp_state2(np2)/real(num_pw,kind=dp)
    end do

    ! Now transform back
    !
    ! Create a plan. We want a backward transform and we want to estimate
    ! (rather than measure) most efficient way to perform the transform.
    plan = fftw_plan_dft_1d(num_pw,tmp_state1,tmp_state2,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! Compute the FFT of in using the current plan, store in out.
    call fftw_execute_dft(plan,tmp_state2,tmp_state1)

    do np2=1,num_pw
      full_H(np2,np1) = tmp_state1(np2)
    end do

  end do

  deallocate(tmp_state2,stat=status)
  if(status/=0) stop 'Error deallocating tmp_stat1 in construct_full_H'

  deallocate(tmp_state1,stat=status)
  if(status/=0) stop 'Error deallocating tmp_stat1 in construct_full_H'

  ! Now add the contribution from the kinetic term
  do np1=1,num_pw
     full_H(np1,np1) = full_H(np1,np1) + cmplx(H_kinetic(np1),0.0_dp,dp)
  end do

  return

end subroutine construct_full_H

subroutine init_random_seed()
  !-------------------------------------------------!
  ! This subroutine sets up the random number       !
  ! generator.                                      !
  !-------------------------------------------------!
  implicit none

  integer                            :: i, n, clock
  integer, dimension(:), allocatable :: seed

  logical, save                      :: initialised=.false.

  if(.not.initialised) then
     call random_seed(size = n)
     allocate(seed(n))

     !call system_clock(count=clock) ! use for non-repeatable random numbers
     clock = 2
     seed = clock + 37 * (/ (i - 1, i = 1, n) /)
     call random_seed(put = seed)

     deallocate(seed)

     initialised = .true.
  end if

  return

end subroutine init_random_seed

subroutine randomise_state(num_pw,num_states,state)
  !-------------------------------------------------!
  ! This subroutine fills the components of a state !
  ! with random numbers.                            !
  !-------------------------------------------------!
  implicit none

  ! Define what we mean by double-precision
  integer, parameter                           :: dp=selected_real_kind(15,300)

  integer,                                     intent(in)    :: num_pw
  integer,                                     intent(in)    :: num_states
  complex(kind=dp), dimension(num_pw,num_states), intent(inout) :: state

  integer                                      :: np
  integer                                      :: nb
  real(kind=dp)                                :: rnd1,rnd2

  ! Set the state randomly s.t. each element lies in [-1,1)
  do nb=1,num_states
     ! 1st plane-wave is G=0 so should be real 
     ! -- with no loss of generality we choose it to be positive as well
     call random_number(rnd1)
     state(1,nb) = cmplx(rnd1,0.0_dp,dp)

     do np=1,num_pw/2
        call random_number(rnd1)
        call random_number(rnd2)
        state(np+1,nb) = 2*cmplx(rnd1-0.5_dp,rnd2-0.5_dp,dp)
        state(num_pw-(np-1),nb) = conjg(state(np+1,nb))
     end do

!     if(2*(num_pw/2)==num_pw) state(num_pw/2,nb) = real(state(num_pw/2,nb),dp)

!     do np=1,num_pw
!        call random_number(rnd1)
!        call random_number(rnd2)
!        state(np,nb) = 2*cmplx(rnd1-0.5_dp,rnd2-0.5_dp,dp)
!     end do
  end do

  return

end subroutine randomise_state


! DQ - For next year we could think about having this in here, i.e. inside the library.
! DQ - this would probably mean having orthonormalise inside the library too which would
! DQ - remove the opportunity for the students to optimise it. I don't think this is an issue.

!!$subroutine line_search(num_pw,num_states,num_proj,approx_state,H_kinetic,H_nonlocal,direction, &
!!$     & gradient,eigenvalue,energy)
!!$  !-------------------------------------------------!
!!$  ! This subroutine takes an approximate eigenstate !
!!$  ! and searches along a direction to find an       !
!!$  ! improved approximation.                         !
!!$  !-------------------------------------------------!
!!$  implicit none
!!$
!!$  ! Define what we mean by double-precision
!!$  integer, parameter                           :: dp=selected_real_kind(15,300)
!!$
!!$  integer,                                     intent(in)    :: num_pw
!!$  integer,                                     intent(in)    :: num_states
!!$  integer,                                     intent(in)    :: num_proj
!!$  real(kind=dp), dimension(num_pw,num_states), intent(inout) :: approx_state
!!$  real(kind=dp), dimension(num_pw),            intent(in)    :: H_kinetic
!!$  real(kind=dp), dimension(num_pw,num_proj),   intent(in)    :: H_nonlocal
!!$  real(kind=dp), dimension(num_pw,num_states), intent(inout) :: direction
!!$  real(kind=dp), dimension(num_pw,num_states), intent(inout) :: gradient
!!$  real(kind=dp),                               intent(inout) :: energy
!!$  real(kind=dp), dimension(num_states),        intent(inout) :: eigenvalue
!!$
!!$  real(kind=dp)                              :: init_energy
!!$  real(kind=dp)                              :: tmp_energy
!!$  !    real(kind=dp), save                        :: trial_step=0.00004_dp
!!$  real(kind=dp), save                        :: trial_step=0.4_dp
!!$  real(kind=dp)                              :: step
!!$  real(kind=dp)                              :: opt_step
!!$  real(kind=dp), dimension(:,:), allocatable :: tmp_state
!!$  real(kind=dp)                              :: d2E_dstep2
!!$  real(kind=dp)                              :: best_step
!!$  real(kind=dp)                              :: best_energy
!!$  real(kind=dp)                              :: denergy_dstep
!!$  real(kind=dp)                              :: mean_norm
!!$  integer                                    :: loop
!!$  integer                                    :: nb
!!$  integer                                    :: status
!!$
!!$  ! To try to keep a convenient step length, we reduce the size of the search direction
!!$  mean_norm = 0.0_dp
!!$  do nb=1,size(approx_state,2)
!!$     mean_norm = mean_norm + sqrt(sum(direction(:,nb)**2))
!!$  end do
!!$  mean_norm = mean_norm/real(size(approx_state,2),dp)
!!$
!!$  direction = direction/mean_norm
!!$
!!$  ! The rate-of-change of the energy is just 2*direction.gradient
!!$  denergy_dstep = 0.0_dp 
!!$  do nb=1,size(approx_state,2)
!!$     denergy_dstep = denergy_dstep + 2*dot_product(direction(:,nb),gradient(:,nb))
!!$  end do
!!$
!!$  allocate(tmp_state(size(approx_state,1),size(approx_state,2)),stat=status)
!!$  if(status/=0) stop 'Error allocating RAM to tmp_state in line_search'
!!$
!!$  best_step   = 0.0_dp
!!$  best_energy = energy
!!$
!!$  ! First take a trial step in the direction
!!$  step = trial_step
!!$
!!$  ! We find a trial step that lowers the energy:
!!$  do loop=1,10
!!$     tmp_state = approx_state + step*direction
!!$
!!$     call orthonormalise(num_pw,num_states,tmp_state)
!!$
!!$     ! Apply the H to this state
!!$     call apply_H(num_pw,num_states,num_proj,tmp_state,H_kinetic,H_nonlocal,gradient)
!!$
!!$     ! Compute the new energy estimate
!!$     tmp_energy = 0.0_dp 
!!$     do nb=1,size(approx_state,2)
!!$        tmp_energy = tmp_energy + dot_product(tmp_state(:,nb),gradient(:,nb))
!!$     end do
!!$
!!$     if(tmp_energy<energy) then
!!$        exit
!!$     else
!!$        d2E_dstep2 = (tmp_energy - energy - step*denergy_dstep )/(step**2)
!!$        if(d2E_dstep2<0.0_dp) then
!!$           if(tmp_energy<energy) then
!!$              exit
!!$           else
!!$              step = step/4.0_dp
!!$           end if
!!$        else
!!$           step  = -denergy_dstep/(2*d2E_dstep2)
!!$        end if
!!$     end if
!!$
!!$  end do
!!$
!!$  if(tmp_energy<best_energy) then
!!$     best_step   = step
!!$     best_energy = tmp_energy
!!$  end if
!!$
!!$  ! We now have the initial eigenvalue, the initial gradient, and a trial step
!!$  ! -- we fit a parabola, and jump to the estimated minimum position
!!$  ! Set default step and energy
!!$  d2E_dstep2 = (tmp_energy - energy - step*denergy_dstep )/(step**2)
!!$
!!$
!!$  if(d2E_dstep2<0.0_dp) then
!!$     ! Parabolic fit gives a maximum, so no good
!!$     write(*,'(a)') '** Warning, parabolic stationary point is a maximum **'
!!$
!!$     if(tmp_energy<energy) then
!!$        opt_step = step
!!$     else
!!$        opt_step = 0.1_dp*step
!!$     end if
!!$  else
!!$     opt_step  = -denergy_dstep/(2*d2E_dstep2)
!!$  end if
!!$
!!$
!!$  !    e = e0 + de*x + c*x**2
!!$  ! => c = (e - e0 - de*x)/x**2
!!$  ! => min. at -de/(2c)
!!$  !
!!$  !    de/dx = de + 2*c*x
!!$
!!$  approx_state = approx_state + opt_step*direction
!!$
!!$  call orthonormalise(num_pw,num_states,approx_state)
!!$
!!$  ! Apply the H to this state
!!$  call apply_H(num_pw,num_states,num_proj,approx_state,H_kinetic,H_nonlocal,gradient)
!!$
!!$  ! Compute the new energy estimate
!!$  energy = 0.0_dp 
!!$  do nb=1,size(approx_state,2)
!!$     eigenvalue(nb) = dot_product(approx_state(:,nb),gradient(:,nb))
!!$     energy = energy + eigenvalue(nb)
!!$  end do
!!$
!!$  ! This ought to be the best, but check...
!!$  if(energy>best_energy) then
!!$     !      if(best_step>0.0_dp) then
!!$     if(abs(best_step-epsilon(1.0_dp))>0.0_dp) then
!!$        approx_state = approx_state + best_step*direction
!!$
!!$        call orthonormalise(num_pw,num_states,approx_state)
!!$
!!$        ! Apply the H to this state
!!$        call apply_H(num_pw,num_states,num_proj,approx_state,H_kinetic,H_nonlocal,gradient)
!!$
!!$        ! Compute the new energy estimate
!!$        energy = 0.0_dp 
!!$        do nb=1,size(approx_state,2)
!!$           eigenvalue(nb) = dot_product(approx_state(:,nb),gradient(:,nb))
!!$           energy = energy + eigenvalue(nb)
!!$        end do
!!$
!!$     else
!!$        write(*,*) 'Oh dear:',best_step
!!$        stop 'Problem with line search'
!!$     end if
!!$  end if
!!$
!!$  !    write(*,'(3f25.15,a)') opt_step,step,energy,' <- test2'
!!$
!!$  ! We'll use this step as the basis of our trial step next time
!!$  !    trial_step = 2*opt_step
!!$
!!$  deallocate(tmp_state,stat=status)
!!$  if(status/=0) stop 'Error deallocating RAM from tmp_state in line_search'
!!$
!!$  return
!!$
!!$end subroutine line_search

