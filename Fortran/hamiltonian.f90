module iterative
  use parameters
  use fft

  implicit none

  contains

    subroutine iterative_solver(H_kinetic, H_local)
      real(kind=dp),dimension(:,:,:), intent(inout) :: H_kinetic, H_local

      real(kind=dp), dimension(:), allocatable :: eigenvalues
      complex(kind=dp), dimension(:,:,:,:), allocatable :: wvfn, gradient

      allocate(wvfn(params%num_plane_waves, params%num_plane_waves, params%num_plane_waves, params%num_states))
      allocate(gradient, mold=wvfn)

      allocate(eigenvalues(params%num_states))

      call randomise_state(wvfn)
      print *, wvfn(:,1,1,1)

      call orthonormalise(wvfn)
      call apply_hamiltonian(H_kinetic, H_local, wvfn, gradient)
      call calculate_eigenvalues(wvfn, gradient, eigenvalues)

      call iterative_search(H_kinetic, H_local, wvfn, gradient, eigenvalues)

      !call diagonalise(wvfn, gradient, eigenvalues)

      !call report_eigenvalues()

      deallocate(eigenvalues)
      deallocate(wvfn)
      deallocate(gradient)

    end subroutine

    subroutine randomise_state(wvfn)
      complex(kind=dp), dimension(:,:,:,:), intent(out) :: wvfn

      real(kind=dp) :: rand_real, rand_imag
      complex(kind=dp) :: random_value
      integer :: x, y, z, state

      integer :: ssize
      integer,dimension(:),allocatable :: seed
      call random_seed(size=ssize)
      allocate(seed(ssize))
      seed = 5
      call random_seed(put=seed)
      deallocate(seed)

      !wvfn = cmplx_0

      do state = 1, params%num_states
        do z = 1, params%num_plane_waves
          do y = 1, params%num_wavevectors+1
            do x = 1, params%num_wavevectors+1
              call random_number(rand_real)

              if(x == 1 .and. y == 1 .and. z == 1) then
                random_value = 2*cmplx(rand_real - 0.5_dp, 0.0_dp, dp)
              else
                call random_number(rand_imag)
                random_value = 2*cmplx(rand_real - 0.5_dp, rand_imag - 0.5_dp, dp)
              endif


              wvfn(x,y,z,state) =  random_value
              if (y .gt. 1) then
                  wvfn(x,params%num_plane_waves-y+2,z,state) = conjg(random_value)
              end if
              if (x .gt. 1) then
                wvfn(params%num_plane_waves-x+2,y,z,state) = -conjg(random_value)
                if (y .gt. 1) then
                  wvfn(params%num_plane_waves-x+2,params%num_plane_waves-y+2,z,state) = -random_value
                end if
              end if

            end do
          end do
        end do
      end do

    end subroutine

    subroutine orthonormalise(wvfn)
      complex(kind=dp), dimension(:,:,:,:), intent(inout) :: wvfn

      complex(kind=dp), dimension(:,:), allocatable :: overlap

      integer :: ns1, ns2, err
      integer :: pw1, pw2, pw3

      !complex :: zdotc

      allocate(overlap(params%num_states,params%num_states))

      do ns2 = 1,params%num_states
        do ns1 = 1,params%num_states

          !overlap(ns1,ns2) = zdotc(params%total_grid_size, conjg(wvfn(:,:,:,ns1)), 1, wvfn(:,:,:,ns2), 1)
          !overlap(ns1,ns2) = 0.0_dp
          !do pw3 = 1,params%num_plane_waves
          !  do pw2 = 1,params%num_plane_waves
          !    do pw1 = 1,params%num_plane_waves
          !      overlap(ns1,ns2) = overlap(ns1,ns2) +  conjg(wvfn(pw1, pw2, pw3, ns1)) * wvfn(pw1, pw2, pw3, ns2)
          !    end do
          !  end do
          !end do
          overlap(ns1,ns2) = sum(conjg(wvfn(:,:,:, ns1)) * wvfn(:, :, :, ns2))

        end do
      end do

      call zpotrf('U', params%num_states, overlap, params%num_states, err)
      if(err .ne. 0) then
        print *, 'zpotrf fail:', err
      endif

      call ztrtri('U', 'N', params%num_states, overlap, params%num_states, err)
      if(err .ne. 0) then
        print *, 'ztrtri fail:', err
      endif

      do ns2 = 1, params%num_states
        do ns1 = ns2+1, params%num_states
          overlap(ns1, ns2) = cmplx_0
        end do
      end do

      call transform(wvfn, overlap)

      deallocate(overlap)

    end subroutine

    subroutine transform(wvfn, operation)
      complex(kind=dp), dimension(:,:,:,:), intent(inout) :: wvfn
      complex(kind=dp), dimension(:,:),     intent(in)    :: operation

      complex(kind=dp), dimension(:,:,:,:), allocatable  :: wvfn_tmp

      integer :: ns1, ns2, pw1, pw2, pw3, stat

      allocate(wvfn_tmp,mold=wvfn)

      wvfn_tmp = cmplx_0

      !print *, operation
      do ns2 = 1,params%num_states
        do ns1 = 1,params%num_states

          wvfn_tmp(:,:,:,ns1) = wvfn_tmp(:,:,:,ns1) + wvfn(:,:,:,ns2) * operation(ns2,ns1)

          !do pw3 = 1,params%num_plane_waves
          !  do pw2 = 1,params%num_plane_waves
          !    do pw1 = 1,params%num_plane_waves
          !      wvfn_tmp(pw1, pw2, pw3, ns1) = wvfn_tmp(pw1, pw2, pw3, ns1) * operation(ns2,ns1)
          !    end do
          !  end do
          !end do

        end do
      end do

      wvfn = wvfn_tmp

      deallocate(wvfn_tmp)
    end subroutine

    subroutine apply_hamiltonian(H_kinetic,H_local,wvfn,gradient)
      complex(kind=dp), dimension(:,:,:,:), intent(inout) :: wvfn, &
                                                             gradient
      real(kind=dp), dimension(:,:,:), intent(in)         :: H_kinetic, &
                                                             H_local

      complex(kind=dp), dimension(:,:,:), allocatable :: tmp_state_out, &
                                                         tmp_state_in


      type(c_ptr) :: plan_forward, plan_backward
      integer :: ns

      allocate(tmp_state_in,mold=wvfn(:,:,:,1))
      allocate(tmp_state_out,mold=wvfn(:,:,:,1))

      plan_forward = fftw_plan_dft_3d(params%num_plane_waves,params%num_plane_waves,params%num_plane_waves, &
        tmp_state_in, tmp_state_out, FFTW_FORWARD, FFTW_ESTIMATE)
      plan_backward = fftw_plan_dft_3d(params%num_plane_waves,params%num_plane_waves,params%num_plane_waves, &
        tmp_state_in, tmp_state_out, FFTW_BACKWARD, FFTW_ESTIMATE)

      ! Make sure gradient is properly zeroed out before adding to it
      gradient = cmplx_0

      do ns=1,params%num_states

        tmp_state_in = wvfn(:,:,:,ns)

        call fftw_execute_dft(plan_forward, tmp_state_in, tmp_state_out)

        tmp_state_in = H_local * tmp_state_out/(params%num_plane_waves*params%num_plane_waves*params%num_plane_waves)

        call fftw_execute_dft(plan_backward, tmp_state_in, tmp_state_out)

        gradient(:,:,:,ns) = tmp_state_out + H_kinetic*wvfn(:,:,:,ns)

      end do

      call fftw_destroy_plan(plan_backward)
      call fftw_destroy_plan(plan_forward)
      print *, wvfn(:,1,1,1)
      print *,'------------'
      print *, gradient(:,1,1,1)

      deallocate(tmp_state_out)
      deallocate(tmp_state_in)
    end subroutine

    subroutine calculate_eigenvalues(wvfn, gradient, eigenvalues)
      complex(kind=dp), dimension(:,:,:,:), intent(in) :: wvfn, gradient
      real(kind=dp), dimension(:), intent(out)         :: eigenvalues

      integer :: ns, pw1, pw2, pw3

      !complex :: zdotc

      do ns=1,params%num_states
        !eigenvalues(ns) = real(zdotc(params%total_grid_size,wvfn(:,:,:,ns),1,gradient(:,:,:,ns),1))
        eigenvalues(ns) = sum(real(conjg(wvfn(:,:,:,ns))*gradient(:,:,:,ns)))

        !do pw3 = 1, params%num_plane_waves
        !  do pw2 = 1, params%num_plane_waves
        !    do pw1 = 1, params%num_plane_waves
        !      eigenvalues(ns) = eigenvalues(ns) + real(conjg(wvfn(pw1,pw2,pw3,ns))*gradient(pw1,pw2,pw3,ns))
        !    end do
        !  end do
        !end do

     end do

    end subroutine

    function check_convergence(previous_energy, total_energy, tolerance) result(converged)
      real(kind=dp) :: previous_energy, total_energy, tolerance
      logical :: converged

      converged = .false.

      if (abs(previous_energy-total_energy) .lt. tolerance) then
        converged = .true.
      endif

    end function

    subroutine orthogonalise(wvfn, gradient)
      complex(kind=dp), dimension(:,:,:,:), intent(in) :: wvfn
      complex(kind=dp), dimension(:,:,:,:), intent(inout) :: gradient

      complex(kind=dp) :: overlap
      integer :: ns1, ns2, pw1,pw2,pw3

      !complex :: zdotc

      do ns2 = 1,params%num_states
        do ns1 = 1,params%num_states
          !overlap = zdotc(params%total_grid_size, wvfn(:,:,:,ns1), 1, gradient(:,:,:,ns2), 1)
          overlap = cmplx_0
          do pw3 = 1,params%num_plane_waves
            do pw2 = 1,params%num_plane_waves
              do pw1 = 1,params%num_plane_waves
                overlap = overlap +  conjg(wvfn(pw1, pw2, pw3, ns1)) * wvfn(pw1, pw2, pw3, ns2)
              end do
            end do
          end do

          gradient(:,:,:,ns2) = gradient(:,:,:,ns2) - overlap*wvfn(:,:,:,ns1)
        end do
      end do

    end subroutine

    subroutine precondition(H_kinetic, wvfn, search_direction)
      complex(kind=dp), dimension(:,:,:,:), intent(in) :: wvfn
      complex(kind=dp), dimension(:,:,:,:), intent(inout) :: search_direction

      real(kind=dp), dimension(:,:,:), intent(in) :: H_kinetic

      real(kind=dp) :: precon, kinetic_eigenvalue, x
      integer :: np1, np2, np3, ns

      do ns = 1,params%num_states

        kinetic_eigenvalue = sum(H_kinetic(:,:,:)*abs(wvfn(:,:,:,ns))**2)

        do np3 = 1,params%num_plane_waves
          do np2 = 1,params%num_plane_waves
            do np1 = 1,params%num_plane_waves

              x = H_kinetic(np1,np2,np3)/kinetic_eigenvalue
              precon = 8.0_dp + x*(4.0_dp + x*( 2.0_dp + 1*x))
              
              search_direction(np1,np2,np3,ns) = search_direction(np1,np2,np2,ns)*precon/(precon+x**4)

            end do
          end do
        end do

      end do
    end subroutine

    subroutine iterative_search(H_kinetic, H_local, wvfn, gradient, eigenvalues)
      complex(kind=dp), dimension(:,:,:,:), intent(inout) :: wvfn, gradient
      
      real(kind=dp), dimension(:,:,:), intent(in) :: H_kinetic, H_local
      real(kind=dp), dimension(:), intent(inout) :: eigenvalues

      complex(kind=dp),dimension(:,:,:,:),allocatable :: search_direction, previous_search_direction
      real(kind=dp) :: gamma, gTxg, previous_gTxg,previous_energy,total_energy,energy_tolerance
      integer :: ns, iter, max_iter, CG_RESET
      logical :: reset_search_direction, converged

      !complex :: zdotc

      integer :: pw1,pw2,pw3
      CG_RESET = 5
      max_iter = 4000
      previous_energy = 1e300_dp
      reset_search_direction = .false.
      converged = .false.
      previous_gTxg = 1.0_dp
      energy_tolerance = 1e-6_dp

      allocate(search_direction, mold=wvfn)
      allocate(previous_search_direction, mold=wvfn)

      previous_energy = 9e300_dp
      total_energy = sum(eigenvalues)
      print *, eigenvalues


      print *, "         | Total Energy | DeltaE"
      print *, " Initial |", total_energy

      do iter = 1, max_iter
        previous_energy = total_energy
        call orthogonalise(wvfn, gradient)

        search_direction = -gradient
        !print *, '------------'
        !print *, search_direction(:,1,1,1)
        !print *, '------------'

        call precondition(H_kinetic, wvfn, search_direction)
        !print *, search_direction(:,1,1,1)
        !print *, '------------'
        call orthogonalise(wvfn, search_direction)
        !print *, search_direction(:,1,1,1)
        !print *, '------------'

        gTxg = 0.0_dp

        do ns = 1, params%num_states
          !gTxg = gTxg + real(zdotc(params%total_grid_size, search_direction(:,:,:,ns), 1, gradient(:,:,:,ns), 1))
          !overlap = cmplx_0
          do pw3 = 1,params%num_plane_waves
            do pw2 = 1,params%num_plane_waves
              do pw1 = 1,params%num_plane_waves
                gTxg = gTxg + real(conjg(search_direction(pw1, pw2, pw3, ns)) * gradient(pw1, pw2, pw3, ns))
              end do
            end do
          end do
        end do

        if (reset_search_direction) then
          gamma = gTxg/previous_gTxg
          search_direction = search_direction + gamma*previous_search_direction
          call orthogonalise(wvfn, search_direction)
          reset_search_direction = .false.
        end if

        previous_gTxg = gTxg

        previous_search_direction = search_direction

        !print *, search_direction(:,1,1,1)
        !print *, '------------'
        print *, 'total energy', total_energy
        call line_search(H_kinetic, H_local, wvfn, search_direction, gradient, eigenvalues, total_energy)
        print *, 'total energy', total_energy

        converged = check_convergence(previous_energy, total_energy, energy_tolerance)

        ! Break out of loop if we're done
        if (converged) then
          print *, 'Converged!'
          exit
        endif

        if (mod(iter,5) .eq. 0) then
          reset_search_direction = .true.
        end if

        print *, iter, total_energy, total_energy-previous_energy

      end do

      deallocate(previous_search_direction)
      deallocate(search_direction)
    end subroutine

    subroutine line_search(H_kinetic, H_local, wvfn, search_direction, gradient, eigenvalues, total_energy)
      real(kind=dp), dimension(:,:,:), intent(in) :: H_kinetic, H_local
      complex(kind=dp), dimension(:,:,:,:), intent(inout) :: wvfn, search_direction, gradient
      real(kind=dp), dimension(:), intent(inout) :: eigenvalues
      real(kind=dp), intent(inout) :: total_energy

      complex(kind=dp), dimension(:,:,:,:), allocatable :: tmp_wvfn

      real(kind=dp) :: epsilon, mean_norm, inv_mean_norm, denergy_dstep, best_step, step, best_energy, &
        & d2E_dstep2, tmp_energy, opt_step
      real(kind=dp) :: trial_step

      integer :: iter,ns

      real(kind=dp) :: dlamch
      complex(kind=dp) :: zdotc

      trial_step = 0.4_dp
      epsilon = dlamch('e')

      mean_norm = sum(abs(search_direction)**2)/params%num_states
      inv_mean_norm = 1.0_dp/mean_norm

      print *, 'abssum', sum(abs(search_direction))
      print *, 'mn:', mean_norm, 'imn:', inv_mean_norm

      search_direction = search_direction * inv_mean_norm

      denergy_dstep = 2*real(zdotc(params%num_states*params%total_grid_size,search_direction,1,gradient,1))
      !denergy_dstep = 2*real(sum(conjg(search_direction)*gradient))
      !print  *, '-----------------------'
      !print *, search_direction(:,1,1,1)
      !print  *, '-----------------------'
      !print *, gradient(:,1,1,1)
      !print  *, '-----------------------'
      !print *, conjg(search_direction(:,1,1,1))*gradient(:,1,1,1)
      !print  *, '-----------------------'

      best_step = 0.0_dp
      best_energy = total_energy
      step = trial_step

      print *, 'denergy_dstep', denergy_dstep
      allocate(tmp_wvfn,mold=wvfn)

      do iter = 1,10
        tmp_wvfn = wvfn + step*search_direction

        call orthonormalise(tmp_wvfn)
        call apply_hamiltonian(H_kinetic,H_local,tmp_wvfn,gradient)

        print *, gradient(:,1,1,1)
        tmp_energy = real(sum(conjg(tmp_wvfn)*gradient))
        print *, 'tmp/tot energy:', tmp_energy, total_energy

        if (tmp_energy .lt. total_energy) then
          print *, 'exit'
          exit
        else
          d2E_dstep2 = (tmp_energy - total_energy - step*denergy_dstep)/(step**2)
          if (d2E_dstep2 .lt. 0.0_dp) then
            step = step/4.0_dp
          else
            step = -denergy_dstep/(2*d2E_dstep2)
          end if
        end if
      end do

      if (tmp_energy .lt. best_energy) then
        best_step = step
        best_energy = tmp_energy
      end if

      d2E_dstep2 = (tmp_energy - total_energy - step*denergy_dstep)/(step**2)
      print *, 'd2:',d2e_dstep2
      if (d2E_dstep2 .lt. 0.0_dp) then
        print *, 'parabolic stationary point is maximum'

        if (tmp_energy .lt. total_energy) then
          opt_step = step
        else
          opt_step = 0.1_dp*step
        end if
      else
        opt_step = -denergy_dstep / (2.0_dp*d2E_dstep2)
      end if

      wvfn = wvfn + opt_step*search_direction

      call orthonormalise(wvfn)
      call apply_hamiltonian(H_kinetic, H_local, wvfn, gradient)

      total_energy = 0.0_dp

      do ns = 1,params%num_states
        !eigenvalues(ns) = real(zdotc(params%total_grid_size,wvfn(:,:,:,ns),1,gradient(:,:,:,ns),1))
        eigenvalues(ns) = real(sum(conjg(wvfn(:,:,:,ns))*gradient(:,:,:,ns)))
        total_energy = total_energy + eigenvalues(ns)
      end do

      if (total_energy .gt. best_energy) then
        if (abs(best_step - epsilon) .gt. 0.0_dp) then
          wvfn = wvfn + best_step * search_direction
          call orthonormalise(wvfn)
          call apply_hamiltonian(H_kinetic, H_local, wvfn, gradient)

          total_energy = 0.0_dp

          do ns = 1,params%num_states
            !eigenvalues(ns) = real(zdotc(params%total_grid_size,wvfn(:,:,:,ns),1,gradient(:,:,:,ns),1))
            eigenvalues(ns) = real(sum(conjg(wvfn(:,:,:,ns))*gradient(:,:,:,ns)))
            total_energy = total_energy + eigenvalues(ns)
          end do
        end if
      end if

      ! TODO this doesn't make sense - should trial_step be 'save'?
      trial_step = 2*opt_step

      deallocate(tmp_wvfn)

    end subroutine

end module
