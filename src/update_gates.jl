## 11/1/2025
## Functions used to update a target two-qubit gate within a set of two-qubit gates
## Use Evenbly-Vidal algorithms to compute the environment tensor and update the target gate



using ITensors
using ITensorMPS
using MKL
using LinearAlgebra
using Random



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



# Define a function to update a single two-qubit gate using Evenbly-Vidal algorithm
function update_single_gate(ψ_L::MPS, ψ_R::MPS, gates_set::Vector{ITensors}, 
  idx::Int64, idx₁::Int64, idx₂::Int64, input_cutoff::Float64 = 1e-10)
  gates_copy = deepcopy(gates_set)
  target = gates_copy[idx]
   
  
  # Remove the target gate from the set of gates and check whether it is removed properly
  deleteat!(gates_copy, idx)
  if target in gates_copy
    error("Target gate was not removed properly before optimization!")
  end

  
  # Apply the gate set without the target gate to the MPS
  ψ_intermediate = apply(gates_copy, ψ_L; cutoff=input_cutoff)
  normalize!(ψ_intermediate)
  i₁, i₂ = siteind(ψ_intermediate, idx₁), siteind(ψ_intermediate, idx₂)
  # @show i₁, i₂
  # println("")


  # Prime specific site indices to compute the environment tensor 
  prime!(ψ_R[idx₁], tags = "Site")
  prime!(ψ_R[idx₂], tags = "Site")
  j₁, j₂ = siteind(ψ_R, idx₁), siteind(ψ_R, idx₂)
  # @show j₁, j₂
  # println("") 


  # Compute the environment tensor from the first site to site idx₁-1
  envL = ITensor(1)
  for j in 1:idx₁-1
    envL *= ψ_intermediate[j]
    envL *= dag(ψ_R[j])
  end


  # Compute the environement tensor from site idx₁+1 to site idx₂-1
  envM = ITensor(1)
  for j in idx₁+1:idx₂-1
    envM *= ψ_intermediate[j]
    envM *= dag(ψ_R[j])
  end 


  # Compute the environment tensor from site idx₂+1 to the last site N
  envR = ITensor(1)
  for j in idx₂ + 1 : N
    envR *= ψ_intermediate[j]
    envR *= dag(ψ_R[j])
  end


  # Compute the environment tensor E from scratch 
  T = ITensor(1)
  T = envL * ψ_intermediate[idx₁] * dag(ψ_R[idx₁])
  T *= envM
  T *= (ψ_intermediate[idx₂] * dag(ψ_R[idx₂]))
  T *= envR
  noprime!(ψ_R)


  # Debugging procedure to make sure the environment tensor is compute correctly
  # @show inds(T)
  # @show inds(target_gate)
  # println("")
  @show tmp_trace = real((T * target)[1])
  @show tmp_cost  = compute_cost_function(ψ_L, ψ_R, gates_set)
  println("")

  
  # Perform SVD on the environment tensor T
  U, S, V = svd(T, (i₁, i₂))
  @show T ≈ U * S * V


  # Commpute the updated two-qubit 
  updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
  
  
  return updated_T, tmp_trace, tmp_cost
end 