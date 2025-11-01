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


const N = 12  # Total number of qubits
const J₁ = 1.0
const τ = 0.5
const cutoff = 1e-12
const nsweeps = 100
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




# Define a function to compute the cost function given two MPS and layers of sets of unitaries
function cost_function_layers(input_ψ_L::MPS, input_ψ_R::MPS, input_gates::Vector{Any}, input_cutoff::Float64 = 1e-10)
  for idx in 1 : length(input_gates)
    layer_of_gates = input_gates[idx]
    input_ψ_L = apply(layer_of_gates, input_ψ_L; cutoff=input_cutoff)
  end
  normalize!(input_ψ_L)

  return real(inner(input_ψ_L, input_ψ_R))
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
  # ψ₀ = MPS(sites, state)                        # Initialize the MPS in a Neel state
  # @show ψ₀

  
  # Measure local observables (one-point functions)
  Sx₀, Sy₀, Sz₀ = zeros(Float64, N), zeros(ComplexF64, N), zeros(Float64, N)
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = -im*expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)


  # Obtain the ground-state wave function of the 1D Heisenberg model via DMRG as the the target MPS
  # Construct the Hamiltonian of the 1D Heisenberg model as an MPO
  os = OpSum()
  for idx = 1:N-1
      os += 0.5 * J₁, "S+", idx, "S-", idx + 1
      os += 0.5 * J₁, "S-", idx, "S+", idx + 1
      os += J₁, "Sz", idx, "Sz", idx + 1
  end

  # Starting from a product state and construct the Hamiltonian as an MPO
  ψ_i = random_mps(sites, state; linkdims=16)
  H = MPO(os, sites)
  
  # Define hyperparameters for DMRG simulation
  nsweeps_dmrg = 10
  maxdim = [20, 50, 200, 2000]
  E, ψ_R = dmrg(H, ψ_i; nsweeps=nsweeps_dmrg, maxdim, cutoff)

  # Measure local observables (one-point functions)
  Sx = expect(ψ_R, "Sx", sites = 1 : N)
  Sz = expect(ψ_R, "Sz", sites = 1 : N)
  println("")
  println("One-point function Sx of the original MPS and target MPS:")
  @show Sx₀
  @show Sx
  println("")
  #*****************************************************************************************************
  #*****************************************************************************************************
  
  
  #*****************************************************************************************************
  #*****************************************************************************************************
  # # Construct a sequence of two-qubit gates as the target unitary operatos
  # target_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
  # gates = ITensor[]
  # for idx in 1 : length(target_pairs)
  #   idx₁, idx₂ = target_pairs[idx][1], target_pairs[idx][2]
  #   s₁ = sites[idx₁]
  #   s₂ = sites[idx₂]

  #   # Define a two-qubit gate, using the Heisenberg interaction as an example 
  #   hj = 1/2 * J₁ * op("S+", s₁) * op("S-", s₂) + 1/2 * J₁ * op("S-", s₁) * op("S+", s₂) + J₁ * op("Sz", s₁) * op("Sz", s₂)
  #   Gj = exp(-im * τ/2 * hj)
  #   push!(gates, Gj)
  # end
  # # @show gates



  # # Construct a sequence of two-qubit gates as the target unitary operatos
  # gates = ITensor[]
  # for idx in 1:2:N-1
  #   idx₁, idx₂ = idx, idx + 1
  #   s₁ = sites[idx₁]
  #   s₂ = sites[idx₂]

  #   # Define a two-qubit gate, using the Heisenberg interaction as an example 
  #   hj = 1/2 * J₁ * op("S+", s₁) * op("S-", s₂) + 1/2 * J₁ * op("S-", s₁) * op("S+", s₂) + J₁ * op("Sz", s₁) * op("Sz", s₂)
  #   Gj = exp(-im * τ/2 * hj)
  #   push!(gates, Gj)
  #   @show inds(Gj)
  # end
  # # @show gates


  # Construct a set of two-qubit gates with random initialization
  indices_pairs = [    
                    [[1, 4], [5, 8], [9, 12]],
                    [[1, 2], [5, 10], [11, 12]], 
                    [[2, 6], [7, 8], [10, 12]], 
                    [[1, 2], [3, 6], [8, 10], [11, 12]]
                  ]
  gates_set = []
  for idx in 1 : length(indices_pairs)
    optimization_gates = ITensor[]
    pairs = indices_pairs[idx]
    
    for index in 1 : length(pairs)
      idx₁, idx₂ = pairs[index][1], pairs[index][2]
      @show idx₁, idx₂
      s₁ = sites[idx₁]
      s₂ = sites[idx₂]
      G_opt = randomITensor(s₁', s₂', s₁, s₂)
      push!(optimization_gates, G_opt)
    end
    # @show optimization_gates
    push!(gates_set, optimization_gates)
  end
  # @show gates_set

  
  # # Apply the sequence of two-qubit gates to the original MPS
  # ψ_R = deepcopy(ψ₀)                        
  # ψ_R = apply(gates, ψ_R; cutoff=cutoff)
  # normalize!(ψ_R)
  
  # # Measure local observables (one-point functions)
  # Sx = expect(ψ_R, "Sx", sites = 1 : N)
  # Sz = expect(ψ_R, "Sz", sites = 1 : N)
  # println("")
  # println("Verify the change in local observables after applying the target two-qubit gates:")
  # @show Sx₀
  # @show Sx
  # println("")
  #*****************************************************************************************************
  #*****************************************************************************************************


  
  #*****************************************************************************************************
  #*****************************************************************************************************
  # Optimize the set of two-qubit gates using an iterative sweeping procedure
  # cost_function, reference = Vector{Float64}(undef, nsweeps), Vector{Float64}(undef, nsweeps)
  cost_function, reference = Float64[], Float64[]
  optimization_trace, fidelity_trace = Float64[], Float64[]
  
  for iteration in 1 : nsweeps
    for layer_idx in 1 : length(gates_set)
      optimization_gates = gates_set[layer_idx]
      pairs = indices_pairs[layer_idx]

      println(repeat("#", 200))
      println("Iteration = $iteration: forward sweep")

      for idx in 1 : length(pairs)
        # Set up the gate set without the target gate
        tmp_Gates = deepcopy(optimization_gates)
        target_gate = tmp_Gates[idx]
        idx₁, idx₂ = pairs[idx][1], pairs[idx][2]
        @show idx₁, idx₂

        # Throw an error if the target gate is still in the temporary gate set
        deleteat!(tmp_Gates, idx)
        if target_gate in tmp_Gates
          error("The gate to be optimized is still in the temporary gate set!")
        end
        
        # # An alternative way to set up the gate set without the target set
        # gate_indices = collect(1 : length(optimization_gates))
        # gate_set_indices = deleteat!(gate_indices, idx)
        # tmp_Gates = ITensor[optimization_gates[i] for i in gate_set_indices]
        # target_gate = optimization_gates[idx]

        ψ_left = ψ₀
        for contraction_idx in 1 : layer_idx - 1
          ψ_left = apply(gates_set[contraction_idx], ψ_left; cutoff=cutoff)
        end
        tmp_ψ = apply(tmp_Gates, ψ_left; cutoff=cutoff)
        normalize!(tmp_ψ)
        i₁, i₂ = siteind(tmp_ψ, idx₁), siteind(tmp_ψ, idx₂)
        
        ψ_right = ψ_R
        for contraction_idx in layer_idx + 1 : length(gates_set)
          ψ_right = apply(gates_set[contraction_idx], ψ_right; cutoff=cutoff)
        end
        normalize!(ψ_right)

        # Set specific site indices to be primed
        prime!(ψ_right[idx₁], tags = "Site")
        prime!(ψ_right[idx₂], tags = "Site")
        j₁, j₂ = siteind(ψ_right, idx₁), siteind(ψ_right, idx₂)
        # @show i₁, i₂, j₁, j₂
        # println("")


        # # Compute the environment tensors from scratch
        # T = ITensor(1)
        # for j in 1:length(tmp_ψ)
        #   T *= (tmp_ψ[j] * dag(ψ_R[j]))
        # end
        # noprime!(ψ_R)
      
        #*****************************************************************************************************
        # Compute the environment tensors using up and down parts
        envL = ITensor(1)
        for j in 1 : idx₁ - 1
          envL *= tmp_ψ[j]
          envL *= dag(ψ_right[j])
          # println("")
          # println("Forward sweep")
          # @show j
          # println("")
        end
        
        
        envM = ITensor(1)
        for j in idx₁ + 1 : idx₂ - 1
          envM *= tmp_ψ[j]
          envM *= dag(ψ_right[j])
          # println("")
          # println("Middle sweep")
          # @show j
          # println("")
        end

        
        envR = ITensor(1)
        for j in idx₂ + 1 : N
          envR *= tmp_ψ[j]
          envR *= dag(ψ_right[j])
          # println("")
          # println("Backward sweep")
          # @show j
          # println("")
        end
        
        
        # @show inds(tmp_ψ)
        # @show inds(ψ_R)
        # @show inds(envL)
        # @show inds(envM)
        # @show inds(envR)

        
        T = ITensor(1)
        T = envL * tmp_ψ[idx₁] * dag(ψ_right[idx₁])
        T *= envM
        T *= (tmp_ψ[idx₂] * dag(ψ_right[idx₂]))
        T *= envR
        noprime!(ψ_right)

        
        # # Compute several traces for debugging purposes
        # envScalar1 = ITensor(1)
        # for env_idx in 1 : N
        #   envScalar1 *= (tmp_ψ[env_idx] * dag(ψ_R[env_idx]))
        # end

        
        # envScalar2 = ITensor(1)
        # envR_copy = deepcopy(envR)
        # noprime!(envR_copy)
        # envScalar2 = envL * envR_copy * (tmp_ψ[2 * idx - 1] * dag(ψ_R[2 * idx - 1])) * (tmp_ψ[2 * idx] * dag(ψ_R[2 * idx]))
        # #**********************************************************************************************************************************************

      
        # @show inds(T)
        # @show inds(target_gate)
        # println("")

        @show real((T * target_gate)[1])
        # @show real(envScalar1[1])
        # @show real(envScalar2[1])
        @show compute_cost_function(ψ_left, ψ_right, optimization_gates)
        # @show compute_cost_function(ψ₀, ψ_R, tmp_Gates, cutoff)
        append!(optimization_trace, real((T * target_gate)[1]))
        append!(fidelity_trace, compute_cost_function(ψ_left, ψ_right, optimization_gates))
        println("")

        
        # Perform SVD (USV†) on the environment tensors
        U, S, V = svd(T, (i₁, i₂))
        @show T ≈ U * S * V
        
        # Update the target two-qubit gate based on Evenbly-Vidal algorithm
        updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
        # @show optimization_gates[idx] == updated_T
        optimization_gates[idx] = updated_T
        # @show optimization_gates[idx] == updated_T
        # @show dag(updated_T) * updated_T
        println("")
      end
      
      
      # Update gates in the backward direction on one sweep
      println(repeat("#", 200))
      println("Iteration = $iteration: backward sweep")
      for idx in length(pairs):-1:1
        # Set up the target gate and the gate set without the target gate
        tmp_Gates = deepcopy(optimization_gates)
        target_gate = tmp_Gates[idx]
        idx₁, idx₂ = pairs[idx][1], pairs[idx][2]
        deleteat!(tmp_Gates, idx)
        if target_gate in tmp_Gates
          error("The gate to be optimized is still in the temporary gate set!")
        end
        

        # Apply the gate set without the target gate and grab all the indices needed to compute the environment tensors 
        ψ_left = ψ₀
        for contraction_idx in 1 : layer_idx - 1
          ψ_left = apply(gates_set[contraction_idx], ψ_left; cutoff=cutoff)
        end
        tmp_ψ = apply(tmp_Gates, ψ_left; cutoff=cutoff)
        normalize!(tmp_ψ)
        i₁, i₂ = siteind(tmp_ψ, idx₁), siteind(tmp_ψ, idx₂)
       
        ψ_right = ψ_R
        for contraction_idx in layer_idx + 1 : length(gates_set)
          ψ_right = apply(gates_set[contraction_idx], ψ_right; cutoff=cutoff)
        end
        normalize!(ψ_right)

        prime!(ψ_right[idx₁], tags = "Site")
        prime!(ψ_right[idx₂], tags = "Site")
        j₁, j₂ = siteind(ψ_right, idx₁), siteind(ψ_right, idx₂)
        # @show i₁, i₂, j₁, j₂
        # println("")


        # Compute the environment tensors from scratch
        T = ITensor(1)
        for j in 1:length(tmp_ψ)
          T *= (tmp_ψ[j] * dag(ψ_right[j]))
        end
        noprime!(ψ_right)


        # Compute trace of the environment tensor times the target gate 
        @show real((T * target_gate)[1])
        @show compute_cost_function(ψ_left, ψ_right, optimization_gates)
        println("")
        

        # Perform SVD (USV†) on the environment tensors
        U, S, V = svd(T, (i₁, i₂))
        @show T ≈ U * S * V      

        # Update the target two-qubit gate based on Evenbly-Vidal algorithm
        updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
        optimization_gates[idx] = updated_T
        println("")
      end
    end


    # Compute and store the cost function after one full sweep
    append!(
      cost_function, 
      cost_function_layers(ψ₀, ψ_R, gates_set, cutoff)
    )
    # cost_function[iteration] = cost_function_layers(ψ₀, ψ_R, gates_set, cutoff)
    # reference[iteration] = compute_cost_function(ψ₀, ψ_R, gates, cutoff)


    # for layer_idx in length(gates_set):-1:1
    #   optimization_gates = gates_set[layer_idx]
    #   pairs = indices_pairs[layer_idx]

    #   println(repeat("#", 200))
    #   println("Iteration = $iteration: forward sweep")

    #   for idx in 1 : length(pairs)
    #     # Set up the gate set without the target gate
    #     tmp_Gates = deepcopy(optimization_gates)
    #     target_gate = tmp_Gates[idx]
    #     idx₁, idx₂ = pairs[idx][1], pairs[idx][2]
    #     @show idx₁, idx₂

    #     # Throw an error if the target gate is still in the temporary gate set
    #     deleteat!(tmp_Gates, idx)
    #     if target_gate in tmp_Gates
    #       error("The gate to be optimized is still in the temporary gate set!")
    #     end
        
    #     # # An alternative way to set up the gate set without the target set
    #     # gate_indices = collect(1 : length(optimization_gates))
    #     # gate_set_indices = deleteat!(gate_indices, idx)
    #     # tmp_Gates = ITensor[optimization_gates[i] for i in gate_set_indices]
    #     # target_gate = optimization_gates[idx]


    #     tmp_ψ = apply(tmp_Gates, ψ₀; cutoff=cutoff)
    #     normalize!(tmp_ψ)
    #     i₁, i₂ = siteind(tmp_ψ, idx₁), siteind(tmp_ψ, idx₂)
        
    #     # Set specific site indices to be primed
    #     prime!(ψ_R[idx₁], tags = "Site")
    #     prime!(ψ_R[idx₂], tags = "Site")
    #     j₁, j₂ = siteind(ψ_R, idx₁), siteind(ψ_R, idx₂)
    #     # @show i₁, i₂, j₁, j₂
    #     # println("")


    #     # # Compute the environment tensors from scratch
    #     # T = ITensor(1)
    #     # for j in 1:length(tmp_ψ)
    #     #   T *= (tmp_ψ[j] * dag(ψ_R[j]))
    #     # end
    #     # noprime!(ψ_R)
      
    #     #*****************************************************************************************************
    #     # Compute the environment tensors using up and down parts
    #     envL = ITensor(1)
    #     for j in 1 : idx₁ - 1
    #       envL *= tmp_ψ[j]
    #       envL *= dag(ψ_R[j])
    #       # println("")
    #       # println("Forward sweep")
    #       # @show j
    #       # println("")
    #     end
        
        
    #     envM = ITensor(1)
    #     for j in idx₁ + 1 : idx₂ - 1
    #       envM *= tmp_ψ[j]
    #       envM *= dag(ψ_R[j])
    #       # println("")
    #       # println("Middle sweep")
    #       # @show j
    #       # println("")
    #     end

        
    #     envR = ITensor(1)
    #     for j in idx₂ + 1 : N
    #       envR *= tmp_ψ[j]
    #       envR *= dag(ψ_R[j])
    #       # println("")
    #       # println("Backward sweep")
    #       # @show j
    #       # println("")
    #     end
        
        
    #     # @show inds(tmp_ψ)
    #     # @show inds(ψ_R)
    #     # @show inds(envL)
    #     # @show inds(envM)
    #     # @show inds(envR)

        
    #     T = ITensor(1)
    #     T = envL * tmp_ψ[idx₁] * dag(ψ_R[idx₁])
    #     T *= envM
    #     T *= (tmp_ψ[idx₂] * dag(ψ_R[idx₂]))
    #     T *= envR
    #     noprime!(ψ_R)

        
    #     # # Compute several traces for debugging purposes
    #     # envScalar1 = ITensor(1)
    #     # for env_idx in 1 : N
    #     #   envScalar1 *= (tmp_ψ[env_idx] * dag(ψ_R[env_idx]))
    #     # end

        
    #     # envScalar2 = ITensor(1)
    #     # envR_copy = deepcopy(envR)
    #     # noprime!(envR_copy)
    #     # envScalar2 = envL * envR_copy * (tmp_ψ[2 * idx - 1] * dag(ψ_R[2 * idx - 1])) * (tmp_ψ[2 * idx] * dag(ψ_R[2 * idx]))
    #     # #**********************************************************************************************************************************************

      
    #     # @show inds(T)
    #     # @show inds(target_gate)
    #     # println("")

    #     @show real((T * target_gate)[1])
    #     # @show real(envScalar1[1])
    #     # @show real(envScalar2[1])
    #     @show compute_cost_function(ψ₀, ψ_R, optimization_gates, cutoff)
    #     # @show compute_cost_function(ψ₀, ψ_R, tmp_Gates, cutoff)
    #     append!(optimization_trace, real((T * target_gate)[1]))
    #     append!(fidelity_trace, compute_cost_function(ψ₀, ψ_R, optimization_gates, cutoff))
    #     println("")

        
    #     # Perform SVD (USV†) on the environment tensors
    #     U, S, V = svd(T, (i₁, i₂))
    #     @show T ≈ U * S * V
        
    #     # Update the target two-qubit gate based on Evenbly-Vidal algorithm
    #     updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
    #     # @show optimization_gates[idx] == updated_T
    #     optimization_gates[idx] = updated_T
    #     # @show optimization_gates[idx] == updated_T
    #     # @show dag(updated_T) * updated_T
    #     println("")
    #   end
      
      
    #   # Update gates in the backward direction on one sweep
    #   println(repeat("#", 200))
    #   println("Iteration = $iteration: backward sweep")
    #   for idx in length(pairs):-1:1
    #     # Set up the target gate and the gate set without the target gate
    #     tmp_Gates = deepcopy(optimization_gates)
    #     target_gate = tmp_Gates[idx]
    #     idx₁, idx₂ = pairs[idx][1], pairs[idx][2]
    #     deleteat!(tmp_Gates, idx)
    #     if target_gate in tmp_Gates
    #       error("The gate to be optimized is still in the temporary gate set!")
    #     end
        

    #     # Apply the gate set without the target gate and grab all the indices needed to compute the environment tensors 
    #     tmp_ψ = apply(tmp_Gates, ψ₀; cutoff=cutoff)
    #     normalize!(tmp_ψ)
    #     i₁, i₂ = siteind(tmp_ψ, idx₁), siteind(tmp_ψ, idx₂)
        
    #     prime!(ψ_R[idx₁], tags = "Site")
    #     prime!(ψ_R[idx₂], tags = "Site")
    #     j₁, j₂ = siteind(ψ_R, idx₁), siteind(ψ_R, idx₂)
    #     # @show i₁, i₂, j₁, j₂
    #     # println("")


    #     # Compute the environment tensors from scratch
    #     T = ITensor(1)
    #     for j in 1:length(tmp_ψ)
    #       T *= (tmp_ψ[j] * dag(ψ_R[j]))
    #     end
    #     noprime!(ψ_R)


    #     # Compute trace of the environment tensor times the target gate 
    #     @show real((T * target_gate)[1])
    #     @show compute_cost_function(ψ₀, ψ_R, optimization_gates)
    #     println("")
        

    #     # Perform SVD (USV†) on the environment tensors
    #     U, S, V = svd(T, (i₁, i₂))
    #     @show T ≈ U * S * V      

        
    #     # Update the target two-qubit gate based on Evenbly-Vidal algorithm
    #     updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
    #     optimization_gates[idx] = updated_T
    #     println("")
    #   end

    #   # Compute and store the cost function after one full sweep
    #   # cost_function[iteration] = compute_cost_function(ψ₀, ψ_R, optimization_gates, cutoff)
    #   # reference[iteration] = compute_cost_function(ψ₀, ψ_R, gates, cutoff)
    # end
    
    # append!(
    #   cost_function, 
    #   cost_function_layers(ψ₀, ψ_R, gates_set, cutoff)
    # )
    # # cost_function[iteration] = cost_function_layers(ψ₀, ψ_R, gates_set, cutoff)
  end

  
  @show cost_function
  # @show reference 
  
  output_filename = "../data/compilation_heisenberg_N$(N)_v2.h5"
  h5open(output_filename, "w") do file
    write(file, "cost function", cost_function)
    write(file, "optimization trace", optimization_trace)
    write(file, "fidelity trace", fidelity_trace)
    # write(file, "reference", reference)
  end
  
  return 
end