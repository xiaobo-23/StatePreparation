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
using ITensors.NDTensors


# Set up parameters for multithreading and parallelization
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


const N = 8  # Total number of qubits
const J₁ = 1.0
const τ = 0.1
const cutoff = 1e-10
const nsweeps = 10
const time_machine = TimerOutput()  # Timing and profiling



let
  #*****************************************************************************************************
  #*****************************************************************************************************
  println(repeat("=", 150))
  println("Optimize two-qubit gates to approximate the time evolution operator")
  
  # Initialize the origiinal random MPS
  sites = siteinds("S=1/2", N; conserve_qns=false)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = random_mps(sites, state; linkdims=20)  # Initialize the original random MPS
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
  # Construc a sequence of two-qubit gates
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
  @show Sz₀
  println("")
  @show Sz_R
  println("")
  
  optimization_gates = ITensor[]
  for idx in 1:2:(N-1)
    s₁ = sites[idx]
    s₂ = sites[idx+1]
    G_opt = randomITensor(s₁', s₂', s₁, s₂)
    push!(optimization_gates, G_opt)
  end
  # @show typeof(optimization_gates)
  # @show length(optimization_gates)
  
  
  cost_function = Vector{Float64}(undef, nsweeps)
  reference_cost_function = Vector{Float64}(undef, nsweeps)
  for iteration in 1 : nsweeps
    for idx in 1 : length(optimization_gates)
      @show optimization_gates[idx]
      
      # Contract psi_L and the two-qubit gate to form a new MPS
      tmp_Gates = deepcopy(optimization_gates)
      deleteat!(tmp_Gates, idx)
      tmp_ψ = apply(tmp_Gates, ψ₀; cutoff=cutoff)
      i₁, i₂ = siteind(tmp_ψ, 2 * idx - 1), siteind(tmp_ψ, 2 * idx)
      @show i₁, i₂
      println("")

      # Set specific site indices to be primed
      # prime!(ψ_R, siteind(ψ_R, 2 * idx - 1))
      # prime!(ψ_R, siteind(ψ_R, 2 * idx))
      prime!(ψ_R[2 * idx - 1], tags = "Site")
      prime!(ψ_R[2 * idx], tags = "Site")
      j₁, j₂ = siteind(ψ_R, 2 * idx - 1), siteind(ψ_R, 2 * idx)
      @show j₁, j₂
      println("")

      T = ITensor()
      for j in 1:length(tmp_ψ)
        if j == 1
          T = tmp_ψ[j] * ψ_R[j]
        else
          T *= tmp_ψ[j] * ψ_R[j]
        end
      end
      # @show inds(tmp_tensor)

      U, S, V = svd(T, i₁, j₁)
      @show T ≈ U * S * V
      # @show S
      S[1, 1] = 1.0
      S[2, 2] = 1.0
      S[3, 3] = 1.0
      S[4, 4] = 1.0
      # @show S

      updated_T = U * S * V
      @show inds(updated_T)
      optimization_gates[idx] = updated_T
      # @show optimization_gates[idx]

      noprime!(ψ_R)
    end

    for idx in length(optimization_gates):-1:1
      @show optimization_gates[idx]
      
      # Contract psi_L and the two-qubit gate to form a new MPS
      tmp_Gates = deepcopy(optimization_gates)
      deleteat!(tmp_Gates, idx)
      tmp_ψ = apply(tmp_Gates, ψ₀; cutoff=cutoff)
      normalize!(tmp_ψ)
      i₁, i₂ = siteind(tmp_ψ, 2 * idx - 1), siteind(tmp_ψ, 2 * idx)
      @show i₁, i₂
      println("")

      # Set specific site indices to be primed
      # prime!(ψ_R, siteind(ψ_R, 2 * idx - 1))
      # prime!(ψ_R, siteind(ψ_R, 2 * idx))
      prime!(ψ_R[2 * idx - 1], tags = "Site")
      prime!(ψ_R[2 * idx], tags = "Site")
      j₁, j₂ = siteind(ψ_R, 2 * idx - 1), siteind(ψ_R, 2 * idx)
      @show j₁, j₂
      println("")

      T = ITensor()
      for j in 1:length(tmp_ψ)
        if j == 1
          T = tmp_ψ[j] * ψ_R[j]
        else
          T *= tmp_ψ[j] * ψ_R[j]
        end
      end
      # @show inds(tmp_tensor)

      U, S, V = svd(T, i₁, j₁)
      @show T ≈ U * S * V
      # @show S
      S[1, 1] = 1.0
      S[2, 2] = 1.0
      S[3, 3] = 1.0
      S[4, 4] = 1.0
      # @show S

      updated_T = U * S * V
      @show inds(updated_T)
      optimization_gates[idx] = updated_T
      # @show optimization_gates[idx]

      noprime!(ψ_R)
    end

    intermidiate_ψ = apply(optimization_gates, ψ₀; cutoff=cutoff)
    normalize!(intermidiate_ψ)
    cost_function[iteration] = real(inner(intermidiate_ψ, ψ_R))




    reference_ψ = apply(gates, ψ₀; cutoff=cutoff)
    normalize!(reference_ψ)
    reference_cost_function[iteration] = real(inner(reference_ψ, ψ_R))

  end

  @show cost_function
  @show reference_cost_function
  return
end