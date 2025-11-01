## 10/7/2025
## Ground state preparation for the Kitaev honeycomb model on cylinders
## Optimize parameters of two-qubit gates

using ITensors
using ITensorMPS
using HDF5
using MKL
using MAT
using LinearAlgebra
using TimerOutputs
using Random


include("update_gates.jl")
include("compute_cost_function.jl")


# Set up parameters for multithreading and parallelization
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


# Define parameters for the compilation 
const N = 12  # Total number of qubits
const J₁ = 1.0
const τ = 0.5
const cutoff = 1e-12
const nsweeps = 2
const time_machine = TimerOutput()  # Timing and profiling


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


  # # Obtain the ground-state wave function of the 1D Heisenberg model via DMRG as the the target MPS
  # # Construct the Hamiltonian of the 1D Heisenberg model as an MPO
  # os = OpSum()
  # for idx = 1:N-1
  #     os += 0.5 * J₁, "S+", idx, "S-", idx + 1
  #     os += 0.5 * J₁, "S-", idx, "S+", idx + 1
  #     os += J₁, "Sz", idx, "Sz", idx + 1
  # end

  # # Starting from a product state and construct the Hamiltonian as an MPO
  # ψ_i = random_mps(sites, state; linkdims=16)
  # H = MPO(os, sites)
  
  # # Define hyperparameters for DMRG simulation
  # nsweeps_dmrg = 10
  # maxdim = [20, 50, 200, 2000]
  # E, ψ_R = dmrg(H, ψ_i; nsweeps=nsweeps_dmrg, maxdim, cutoff)

  # # Measure local observables (one-point functions)
  # Sx = expect(ψ_R, "Sx", sites = 1 : N)
  # Sz = expect(ψ_R, "Sz", sites = 1 : N)
  # println("")
  # println("One-point function Sx of the original MPS and target MPS:")
  # @show Sx₀
  # @show Sx
  # println("")
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



  # Construct a sequence of two-qubit gates as the target unitary operators
  gates = ITensor[]
  for idx in 1:2:N-1
    idx₁, idx₂ = idx, idx + 1
    s₁ = sites[idx₁]
    s₂ = sites[idx₂]

    # Define a two-qubit gate, using the Heisenberg interaction as an example 
    hj = 1/2 * J₁ * op("S+", s₁) * op("S-", s₂) + 1/2 * J₁ * op("S-", s₁) * op("S+", s₂) + J₁ * op("Sz", s₁) * op("Sz", s₂)
    Gj = exp(-im * τ/2 * hj)
    push!(gates, Gj)
    @show inds(Gj)
  end
  # @show gates


  # Construct a set of two-qubit gates with random initialization
  # indices_pairs = [    
  #                   # [[1, 4], [5, 8], [9, 12]],
  #                   [[1, 2], [5, 10], [11, 12]], 
  #                   [[2, 6], [7, 8], [10, 12]], 
  #                   [[1, 2], [3, 6], [8, 10], [11, 12]]
  #                 ]
  indices_pairs = [
                    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
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


  # Apply the sequence of two-qubit gates to the original MPS
  ψ_R = deepcopy(ψ₀)
  ψ_R = apply(gates, ψ_R; cutoff=cutoff)
  normalize!(ψ_R)
  
  # Measure local observables (one-point functions)
  Sx = expect(ψ_R, "Sx", sites = 1 : N)
  Sz = expect(ψ_R, "Sz", sites = 1 : N)
  println("")
  println("Verify the change in local observables after applying the target two-qubit gates:")
  @show Sx₀
  @show Sx
  println("")
  #*****************************************************************************************************
  #*****************************************************************************************************


  
  #*****************************************************************************************************
  #*****************************************************************************************************
  # Optimize the set of two-qubit gates using an iterative sweeping procedure
  cost_function, reference = Float64[], Float64[]
  optimization_history, trace_history = Float64[], Float64[]
  
  for iteration in 1 : nsweeps
    for layer_idx in 1 : length(gates_set)
      println(repeat("#", 200))
      println("Optimizing layered two-qubit gate in the backward direction @ layer $layer_idx")
      println("")

      optimization_gates = gates_set[layer_idx]
      pairs = indices_pairs[layer_idx]

      # Update each two-qubit gate in forward direction 
      println(repeat("#", 200))
      println("Iteration = $iteration: forward sweep")

      for idx in 1:length(pairs)
        # Update each two-qubit gate using Evenbly-Vidal algorithm
        optimization_gates[idx], tmp_trace, tmp_cost = update_single_gate(
          ψ₀, 
          ψ_R, 
          optimization_gates, 
          idx, 
          pairs[idx][1], 
          pairs[idx][2], 
          cutoff
        )
        
        # Store traces and cost function values for analysis
        append!(trace_history, tmp_trace)
        append!(optimization_history, tmp_cost)

        # ψ_left = ψ₀
        # for contraction_idx in 1 : layer_idx - 1
        #   ψ_left = apply(gates_set[contraction_idx], ψ_left; cutoff=cutoff)
        # end
        # tmp_ψ = apply(tmp_Gates, ψ_left; cutoff=cutoff)
        # normalize!(tmp_ψ)
        # i₁, i₂ = siteind(tmp_ψ, idx₁), siteind(tmp_ψ, idx₂)
        
        # ψ_right = ψ_R
        # for contraction_idx in length(gates_set):-1:layer_idx + 1
        #   ψ_right = apply(gates_set[contraction_idx], ψ_right; cutoff=cutoff)
        # end
        # normalize!(ψ_right)

        # # Set specific site indices to be primed
        # prime!(ψ_right[idx₁], tags = "Site")
        # prime!(ψ_right[idx₂], tags = "Site")
        # j₁, j₂ = siteind(ψ_right, idx₁), siteind(ψ_right, idx₂)
        # # @show i₁, i₂, j₁, j₂
        # # println("")
      end
      println("")
      
      # Update gates in the backward direction
      println(repeat("#", 200))
      println("Iteration = $iteration: backward sweep")
      
      for idx in length(pairs):-1:1
        # Update each two-qubit gate using Evenbly-Vidal algorithm
        optimization_gates[idx], tmp_trace, tmp_cost = update_single_gate(
          ψ₀, 
          ψ_R, 
          optimization_gates, 
          idx, 
          pairs[idx][1], 
          pairs[idx][2], 
          cutoff
        )

        # Store traces and cost function values for analysis
        append!(trace_history, tmp_trace)
        append!(optimization_history, tmp_cost)
      end
    
      
      # Compute and store the cost function after one full sweep of a specific layer 
      append!(
        cost_function, 
        cost_function_layers(ψ₀, ψ_R, gates_set, cutoff)
      )
    end
    println("")



    # Optimize two-qubit gates layer by layer in the backward direction if the number of layers is larger than one
    if length(gates_set) > 1
      for layer_idx in length(gates_set):-1:1
        println(repeat("#", 200))
        println("Optimizing layered two-qubit gate in the backward direction @ layer $layer_idx")
        println("")
        
        optimization_gates = gates_set[layer_idx]
        pairs = indices_pairs[layer_idx]

        # Update gates in one single layer in the forward direction 
        println(repeat("#", 200))
        println("Iteration = $iteration: forward sweep")

        for idx in 1 : length(pairs)
          optimization_gates[idx], tmp_trace, tmp_cost = update_single_gate(
            ψ₀, 
            ψ_R, 
            optimization_gates, 
            idx, 
            pairs[idx][1], 
            pairs[idx][2], 
            cutoff
          )

          # Store traces and cost function values for analysis
          append!(trace_history, tmp_trace)
          append!(optimization_history, tmp_cost)
        end
        println("")
        
        # Update gates in one single layer in the backward direction 
        println(repeat("#", 200))
        println("Iteration = $iteration: backward sweep")
        for idx in length(pairs):-1:1
          optimization_gates[idx], tmp_trace, tmp_cost = update_single_gate(
            ψ₀, 
            ψ_R, 
            optimization_gates, 
            idx, 
            pairs[idx][1], 
            pairs[idx][2], 
            cutoff
          )

          # Store traces and cost function values for analysis
          append!(trace_history, tmp_trace)
          append!(optimization_history, tmp_cost)
        end
        println("")


        # Compute and store the cost function after one full sweep of a specific layer 
        append!(
          cost_function, 
          cost_function_layers(ψ₀, ψ_R, gates_set, cutoff)
        )
      end
    end    
  end

  
  # Visualize and output the history of the cost function 
  println("Final cost function after optimization:")
  @show cost_function
  println(repeat("#", 200))
  println(repeat("#", 200))


  # output_filename = "../data/compilation_heisenberg_N$(N)_v2.h5"
  # h5open(output_filename, "w") do file
  #   write(file, "cost function", cost_function)
  #   write(file, "optimization trace", optimization_trace)
  #   write(file, "fidelity trace", fidelity_trace)
  #   # write(file, "reference", reference)
  # end
  
  return 
end