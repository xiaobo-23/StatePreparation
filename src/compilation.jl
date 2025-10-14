## 10/7/2025
## Ground state preparation for the Kitaev honeycomb model on cylinders
## Optimize parameters of two-qubit gates

using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs
using MAT
using Random


# Set up parameters for multithreading and parallelization
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


const N = 8  # Total number of qubits
const J₁ = 1.0
const τ = 0.5
const cutoff = 1e-10
const nsweeps = 1
const time_machine = TimerOutput()  # Timing and profiling


# Define a function to compute the cost function given two MPS and a set of unitaries
function compute_cost_function(input_ψ_L::MPS, input_ψ_R::MPS, input_gates::Vector{ITensor})
  intermediate_psi = apply(input_gates, input_ψ_L; cutoff=cutoff)
  # normalize!(intermediate_psi)
  return real(inner(intermediate_psi, input_ψ_R))
end



let
  #*****************************************************************************************************
  #*****************************************************************************************************
  println(repeat("#", 200))
  println("Optimize two-qubit gates to approximate the time evolution operator")
  
  
  Random.seed!(123)
  # Initialize the origiinal random MPS
  sites = siteinds("S=1/2", N; conserve_qns=false)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = random_mps(sites, state; linkdims=16)  # Initialize the original random MPS
  # ψ₀ = MPS(sites, state)                    # Initialize the MPS in a Neel state
  # @show ψ₀

  # Measure local observables (one-point functions)
  Sx₀, Sy₀, Sz₀ = zeros(Float64, N), zeros(ComplexF64, N), zeros(Float64, N)
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = -im*expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  # @show Sx₀
  # println("")
  # @show Sy₀
  # println("")
  # @show Sz₀ 
  #*****************************************************************************************************
  #*****************************************************************************************************
  
  
  #*****************************************************************************************************
  #*****************************************************************************************************
  # Construct a sequence of two-qubit gates
  gates = ITensor[]
  for idx in 3:2:(N-1)
    # @show idx
    s₁ = sites[idx]
    s₂ = sites[idx+1]

    # Define a two-qubit gate, using the Heisenberg interaction as an example 
    hj = 1/2 * J₁ * op("S+", s₁) * op("S-", s₂) + 1/2 * J₁ * op("S-", s₁) * op("S+", s₂) + J₁ * op("Sz", s₁) * op("Sz", s₂)
    Gj = exp(-im * τ/2 * hj)
    push!(gates, Gj)
  end
  # @show gates


  ψ_R = deepcopy(ψ₀)                        # Create a copy of the original MPS to apply gates
  ψ_R = apply(gates, ψ_R; cutoff=cutoff)
  normalize!(ψ_R)
  
  Sx_R, Sz_R = zeros(Float64, N), zeros(Float64, N)
  Sx_R = expect(ψ_R, "Sx", sites = 1 : N)
  Sz_R = expect(ψ_R, "Sz", sites = 1 : N)
  println("")
  println("After applying two-qubit gates:")
  println("")
  @show Sx₀
  println("")
  @show Sx_R
  println("")
  # @show Sz₀
  # println("")
  # @show Sz_R
  # println("")
  
  
  optimization_gates = ITensor[]
  for idx in 3:2:(N-1)
    s₁ = sites[idx]
    s₂ = sites[idx+1]
    G_opt = randomITensor(s₁', s₂', s₁, s₂)
    push!(optimization_gates, G_opt)
  end
  # @show optimization_gates
  # @show typeof(optimization_gates)
  # @show length(optimization_gates)
  set_copy = deepcopy(optimization_gates)


  
  cost_function, reference = Vector{Float64}(undef, nsweeps), Vector{Float64}(undef, nsweeps)
  for iteration in 1 : nsweeps
    println(repeat("#", 200))
    println("Iteration = $iteration: forward sweep")
    for idx in 1 : length(optimization_gates)
      # Contract psi_L and the two-qubit gate to form a new MPS
      tmp_Gates = deepcopy(optimization_gates)
      gate₀ = tmp_Gates[idx]
      # if idx != 1
      #   @show optimization_gates[idx - 1]
      # end
      # @show set_copy[idx - 1]
      deleteat!(tmp_Gates, idx)
  

      tmp_ψ = apply(tmp_Gates, ψ₀; cutoff=cutoff)
      # normalize!(tmp_ψ)
      i₁, i₂ = siteind(tmp_ψ, 2 * idx + 1), siteind(tmp_ψ, 2 * idx + 2)
      # @show i₁, i₂
      # println("")

      # Set specific site indices to be primed
      # prime!(ψ_R, siteind(ψ_R, 2 * idx - 1))
      # prime!(ψ_R, siteind(ψ_R, 2 * idx))
      prime!(ψ_R[2 * idx + 1], tags = "Site")
      prime!(ψ_R[2 * idx + 2], tags = "Site")
      j₁, j₂ = siteind(ψ_R, 2 * idx + 1), siteind(ψ_R, 2 * idx + 2)
      # @show j₁, j₂
      # println("")

      
      # # Compute the environment tensors from scratch
      # T = ITensor(1)
      # for j in 1:length(tmp_ψ)
      #   T *= (tmp_ψ[j] * ψ_R[j])
      # end
     
      #*****************************************************************************************************
      # Compute the environment tensors using up and down parts
      envL = ITensor(1)
      for j in 1:(2 * idx - 2)
        if j == 1
          envL = tmp_ψ[j]
          envL *= ψ_R[j]
        else
          envL *= tmp_ψ[j]
          envL *= ψ_R[j]
        end
      end

      envR = ITensor(1)
      for j in N:-1:(2 * idx + 1)
        if j == N
          envR = tmp_ψ[j]
          envR *= ψ_R[j]
        else
          envR *= tmp_ψ[j]
          envR *= ψ_R[j]
        end
      end

      T = ITensor()
      T = envR * tmp_ψ[2 * idx - 1] * ψ_R[2 * idx - 1]
      T *= tmp_ψ[2 * idx] * ψ_R[2 * idx] 
      T *= envL
      #*****************************************************************************************************
      noprime!(ψ_R)

      # @show T
      # @show inds(T)
      # println("")
      
      # # @show gate₀
      # @show inds(gate₀)
      # println("")
      
      # tmpU, tmpS, tmpV = svd(T, i₁, i₂)
      # T_transpose = dag(tmpV) * tmpS * dag(tmpU)
     
      gates_set = push!(deepcopy(tmp_Gates), gate₀)
      @show real((T * gate₀))[1]
      @show compute_cost_function(ψ₀, ψ_R, optimization_gates)
      @show compute_cost_function(ψ₀, ψ_R, gates_set)
      println("")

      
      # U, S, V = svd(T, (i₁, j₁))
      U, S, V = svd(T, (i₁, i₂))
      @show T ≈ U * S * V
      # @show inds(U)
      
     
      # Setting the singular values to be 1 by hands
      # # @show S
      # S[1, 1] = 1.0
      # S[2, 2] = 1.0
      # S[3, 3] = 1.0
      # S[4, 4] = 1.0
      # # @show S
      # updated_T = dag(V) * S * dag(U)
      
      
      # TO-DO: figure out the row indices used in svd
      updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
      optimization_gates[idx] = dag(updated_T)
      
      # @show dag(updated_T) * updated_T
      println("")
      # @show updated_T
      # @show optimization_gates[idx]
    end
    println("")
    
    println(repeat("#", 200))
    println("Iteration = $iteration: backward sweep")
    for idx in length(optimization_gates)-1:-1:1
      # @show optimization_gates[idx]
      
      # Contract psi_L and the two-qubit gate to form a new MPS
      tmp_Gates = deepcopy(optimization_gates)
      gate₀ = tmp_Gates[idx]
      deleteat!(tmp_Gates, idx)
      tmp_ψ = apply(tmp_Gates, ψ₀; cutoff=cutoff)
      # normalize!(tmp_ψ)
      i₁, i₂ = siteind(tmp_ψ, 2 * idx + 1), siteind(tmp_ψ, 2 * idx + 2)
      # @show i₁, i₂
      # println("")

      # Set specific site indices to be primed
      # prime!(ψ_R, siteind(ψ_R, 2 * idx - 1))
      # prime!(ψ_R, siteind(ψ_R, 2 * idx))
      prime!(ψ_R[2 * idx + 1], tags = "Site")
      prime!(ψ_R[2 * idx + 2], tags = "Site")
      j₁, j₂ = siteind(ψ_R, 2 * idx + 1), siteind(ψ_R, 2 * idx + 2)
      # @show j₁, j₂
      # println("")


      T = ITensor()
      for j in 1:length(tmp_ψ)
        if j == 1
          T = tmp_ψ[j] * ψ_R[j]
        else
          T *= (tmp_ψ[j] * ψ_R[j])
        end
      end
      noprime!(ψ_R)

      
      @show real((T * dag(gate₀)))[1]
      @show compute_cost_function(ψ₀, ψ_R, optimization_gates)
      println("")

      
      # U, S, V = svd(T, (i₁, j₁))
      U, S, V = svd(T, (i₁, i₂))
      # @show T ≈ U * S * V

      # # @show S
      # S[1, 1] = 1.0
      # S[2, 2] = 1.0
      # S[3, 3] = 1.0
      # S[4, 4] = 1.0
      # # @show S
      
      
      # updated_T = U * S * V
      # updated_T = dag(V) * S * dag(U)
      updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
      optimization_gates[idx] = updated_T
      # @show dag(updated_T) * updated_T
      # @show inds(updated_T)
      println("")
    end
    println("")

    # Compute the cost function after each sweep
    cost_function[iteration] = compute_cost_function(ψ₀, ψ_R, optimization_gates)
    reference[iteration] = compute_cost_function(ψ₀, ψ_R, gates)
  end

  @show cost_function
  @show reference 
  
  # output_filename = "compilation_N$(N)_v3.h5"
  # h5open(output_filename, "w") do file
  #   write(file, "cost function", cost_function)
  #   write(file, "reference", reference)
  # end
  
  return
end