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


const N = 8 # Total number of qubits
const J₁ = 1.0
const τ = 0.5
const cutoff = 1e-12
const nsweeps = 2
const time_machine = TimerOutput()  # Timing and profiling


# Define a function to compute the cost function given two MPS and a set of unitaries
function compute_cost_function(input_ψ_L::MPS, input_ψ_R::MPS, input_gates::Vector{ITensor}, input_cutoff::Float64 = 1e-10)
  intermediate_psi = apply(input_gates, input_ψ_L; cutoff=input_cutoff)
  normalize!(intermediate_psi)

  # fidelity = ITensor(1)
  # for idx₁ in 1:length(intermediate_psi)
  #   fidelity *= (intermediate_psi[idx₁] * dag(input_ψ_R[idx₁]))
  # end
  # @show real(fidelity[1]) ≈ real(inner(intermediate_psi, input_ψ_R)), real(fidelity[1]), real(inner(intermediate_psi, input_ψ_R)) 
  return real(inner(intermediate_psi, input_ψ_R))
end



let
  #*****************************************************************************************************
  #*****************************************************************************************************
  println(repeat("#", 200))
  println("Optimize two-qubit gates to approximate the time evolution operator")
  
  
  Random.seed!(1234567)
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
  for idx in 1:2:(N-1)
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
  for idx in 1:2:(N-1)
    s₁ = sites[idx]
    s₂ = sites[idx+1]
    G_opt = randomITensor(s₁', s₂', s₁, s₂)
    push!(optimization_gates, G_opt)
  end
  # @show optimization_gates
  # @show typeof(optimization_gates)
  # @show length(optimization_gates)
  

  cost_function, reference = Vector{Float64}(undef, nsweeps), Vector{Float64}(undef, nsweeps)
  optimization_trace, fidelity_trace = Float64[], Float64[]
  for iteration in 1 : nsweeps
    println(repeat("#", 200))
    println("Iteration = $iteration: forward sweep")

    
    for idx in 1 : length(optimization_gates)
      # # Set up the gate set without the target gate
      # tmp_Gates = deepcopy(optimization_gates)
      # target_gate = tmp_Gates[idx]
      # # @show target_gate

      
      # # @show length(tmp_Gates)
      # deleteat!(tmp_Gates, idx)
      # if target_gate in tmp_Gates
      #   error("The gate to be optimized is still in the temporary gate set!")
      # end
      # # @show tmp_Gates
      # # @show length(tmp_Gates)


      # An alternative way to set up the gate set without the target set
      gate_indices = collect(1 : length(optimization_gates))
      gate_set_indices = deleteat!(gate_indices, idx)
      tmp_Gates = ITensor[optimization_gates[i] for i in gate_set_indices]
      target_gate = optimization_gates[idx]


      # @show expect(ψ₀, "Sz", sites = 1 : length(ψ₀)) 
      tmp_ψ = apply(tmp_Gates, ψ₀; cutoff=cutoff)
      normalize!(tmp_ψ)
      i₁, i₂ = siteind(tmp_ψ, 2 * idx - 1), siteind(tmp_ψ, 2 * idx)
      # @show i₁, i₂
      # println("")

      
      # Set specific site indices to be primed
      # prime!(ψ_R, siteind(ψ_R, 2 * idx - 1))
      # prime!(ψ_R, siteind(ψ_R, 2 * idx))
      prime!(ψ_R[2 * idx - 1], tags = "Site")
      prime!(ψ_R[2 * idx], tags = "Site")
      j₁, j₂ = siteind(ψ_R, 2 * idx - 1), siteind(ψ_R, 2 * idx)
      # @show j₁, j₂
      # println("")


      # @show expect(ψ₀, "Sz", sites = 1 : length(ψ₀))
      test₁ = apply([tmp_Gates[1]], ψ₀; cutoff=cutoff)
      normalize!(test₁)
      
      test₂ = apply(optimization_gates, ψ₀; cutoff=cutoff)
      normalize!(test₂)

      # # Compute the environment tensors from scratch
      # T = ITensor(1)
      # for j in 1:length(tmp_ψ)
      #   T *= (tmp_ψ[j] * dag(ψ_R[j]))
      # end
     
      #*****************************************************************************************************
      # Compute the environment tensors using up and down parts
      envL = ITensor(1)
      for j in 1:(2 * idx - 2)
        envL *= tmp_ψ[j]
        envL *= dag(ψ_R[j])
        println("")
        println("Forward sweep")
        @show j
        println("")
      end


      envR = ITensor(1)
      for j in N:-1:(2 * idx + 1)
        envR *= tmp_ψ[j]
        envR *= dag(ψ_R[j])
      
        println("")
        println("Backward sweep")
        @show j
        println("")
      end


      # T = ITensor()
      # T = envL * tmp_ψ[2 * idx - 1] * dag(ψ_R[2 * idx - 1])
      # T *= (tmp_ψ[2 * idx] * dag(ψ_R[2 * idx]))
      # T *= envR
      # # @show inds(T)
      # noprime!(ψ_R)

      
      # Compute the environment tensors from scratch
      T = ITensor(1)
      for idx in 1:N
        T *= (tmp_ψ[idx] * dag(ψ_R[idx]))
      end
      noprime!(ψ_R)


      # Compute several traces for debugging purposes
      envScalar1 = ITensor(1)
      for env_idx in 1 : N
        envScalar1 *= (tmp_ψ[env_idx] * dag(ψ_R[env_idx]))
      end

      
      envScalar2 = ITensor(1)
      envR_copy = deepcopy(envR)
      noprime!(envR_copy)
      envScalar2 = envL * envR_copy * (tmp_ψ[2 * idx - 1] * dag(ψ_R[2 * idx - 1])) * (tmp_ψ[2 * idx] * dag(ψ_R[2 * idx]))

      
      envScalar3 = ITensor(1)
      for env_idx in 1 : N
        envScalar3 *= (test₁[env_idx] * dag(ψ_R[env_idx]))
      end

      
      envScalar4 = ITensor(1)
      for env_idx in 1 : N
        envScalar4 *= (test₂[env_idx] * dag(ψ_R[env_idx]))
      end
      #*************************************************************************************************************************

    
      @show inds(T)
      @show inds(target_gate)
      println("")

     
      # @show real((target_gate * dag(T))[1])
      @show real((T * target_gate)[1])
      @show real(envScalar1[1])
      @show real(envScalar2[1])
      @show real(envScalar3[1])
      @show real(envScalar4[1])
      @show compute_cost_function(ψ₀, ψ_R, optimization_gates, cutoff)
      @show compute_cost_function(ψ₀, ψ_R, tmp_Gates, cutoff)
      @show compute_cost_function(ψ₀, ψ_R, [tmp_Gates[1]], cutoff)
      append!(optimization_trace, real((T * target_gate)[1]))
      append!(fidelity_trace, compute_cost_function(ψ₀, ψ_R, optimization_gates, cutoff))
      println("")

      
      # Use Evenbly-Vidal method to perform SVD on the environment tensors
      U, S, V = svd(T, (i₁, i₂))
      @show T ≈ U * S * V
      

      # Setting the singular values to 1
      # # @show S
      # S[1, 1] = 1.0
      # S[2, 2] = 1.0
      # S[3, 3] = 1.0
      # S[4, 4] = 1.0
      # # @show S
      # updated_T = dag(V) * S * dag(U)
      
      
      # Update the target two-qubit gate 
      updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
      # @show optimization_gates[idx]
      @show optimization_gates[idx] == updated_T
      optimization_gates[idx] = updated_T
      # @show dag(updated_T) * updated_T
      # @show optimization_gates[idx]
      @show optimization_gates[idx] == updated_T
      println("")
    end
    println("")
    
    
    # Backward propagation in one sweep
    println(repeat("#", 200))
    println("Iteration = $iteration: backward sweep")
    for idx in length(optimization_gates):-1:1
      # Set up the target gate and the gate set without the target gate
      tmp_Gates = deepcopy(optimization_gates)
      target_gate = tmp_Gates[idx]
      deleteat!(tmp_Gates, idx)
      if target_gate in tmp_Gates
        error("The gate to be optimized is still in the temporary gate set!")
      end
      

      # Apply the gate set without the target gate and grab all the indices needed to compute the environment tensors 
      tmp_ψ = apply(tmp_Gates, ψ₀; cutoff=cutoff)
      normalize!(tmp_ψ)
      i₁, i₂ = siteind(tmp_ψ, 2 * idx - 1), siteind(tmp_ψ, 2 * idx)
      
      prime!(ψ_R[2 * idx - 1], tags = "Site")
      prime!(ψ_R[2 * idx], tags = "Site")
      j₁, j₂ = siteind(ψ_R, 2 * idx - 1), siteind(ψ_R, 2 * idx)
      # @show i₁, i₂, j₁, j₂
      # println("")


      # Compute the environment tensors from scratch
      T = ITensor(1)
      for j in 1:length(tmp_ψ)
        T *= (tmp_ψ[j] * dag(ψ_R[j]))
      end
      noprime!(ψ_R)


      # Compute trace of the environment tensor times the target gate 
      @show real((T * target_gate)[1])
      @show compute_cost_function(ψ₀, ψ_R, optimization_gates)
      println("")
      

      # Use Evenbly-Vidal algorithm to perform SVD on the environment tensors
      U, S, V = svd(T, (i₁, i₂))
      @show T ≈ U * S * V      

      # Update the target two-qubit gate
      updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
      # @show optimization_gates[idx] == updated_T
      optimization_gates[idx] = updated_T
      # @show optimization_gates[idx] == updated_T
      println("")

    end
    println("")



    # Compute the cost function after each sweep
    cost_function[iteration] = compute_cost_function(ψ₀, ψ_R, optimization_gates, cutoff)
    reference[iteration] = compute_cost_function(ψ₀, ψ_R, gates, cutoff)
  end

  @show cost_function
  @show reference 
  
  # output_filename = "compilation_N$(N)_v2.h5"
  # h5open(output_filename, "w") do file
  #   write(file, "cost function", cost_function)
  #   write(file, "reference", reference)
  #   write(file, "optimization trace", optimization_trace)
  #   write(file, "fidelity trace", fidelity_trace)
  # end
  
  return
end 