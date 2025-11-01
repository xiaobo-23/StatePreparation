## 11/1/2025
## Compute the cost function given a set of two-qubit gates and two matrix product states
## Works for both a single layer of two-qubit gates and multiple layers of two-qubit gates  

using ITensors
using ITensorMPS


# Define the function to compute the cost function using two matrix product states 
# and a single layer of two-qubit gates as input
function compute_cost_function(psi_L::MPS, psi_R::MPS, input_gates::Vector{ITensor}, 
  input_cutoff::Float64 = 1e-10)
  psi_intermediate = apply(input_gates, psi_L; cutoff=input_cutoff)
  normalize!(psi_intermediate)

  # fidelity = ITensor(1)
  # for idx₁ in 1:length(psi_intermediate)
  #   fidelity *= (psi_intermediate[idx₁] * dag(psi_R[idx₁]))
  # end
  # @show real(fidelity[1]) ≈ real(inner(psi_intermediate, psi_R)), real(fidelity[1]), real(inner(psi_intermediate, psi_R)) 
  
  return real(inner(psi_intermediate, psi_R))
end


# Define the function to compute the cost function using two matrix product states
# and multiple layers of two-qubit gates as input
function cost_function_layers(psi_L::MPS, psi_R::MPS, input_gates::Vector{Any}, 
  input_cutoff::Float64 = 1e-10)

  circuit_depth = length(input_gates)

  for idx in 1 : circuit_depth
    psi_L = apply(input_gates[idx], psi_L; cutoff=input_cutoff)
  end
  normalize!(psi_L)

  return real(inner(psi_L, psi_R))
end