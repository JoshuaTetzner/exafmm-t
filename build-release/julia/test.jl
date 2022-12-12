
path = "/home/jt286/Documents/Code/C++/exafmm-t/build-release/julia/libExaFMMCInterface"
##
ns=20
p=8
ncrit=50
a = rand(Cdouble, 3, 3)#[rand(Cdouble, 3) for i = 1:ns] 
a = [x for x in eachrow(a)];
b = ComplexF64.(rand(Float64, ns))
wavek = ComplexF64(0)

##
src = ccall((:init_sources_C64, path), Ptr{Cvoid}, (Ptr{Ptr{Cdouble}}, Ptr{ComplexF64}, Int64), a, b, ns)
trg = ccall((:init_targets_C64, path), Ptr{Cvoid}, (Ptr{Ptr{Cdouble}}, Int64), a, ns)
##
fmm = ccall((:HelmholtzFMM, path), Ptr{Cvoid}, (Cint, Cint, ComplexF64), p, ncrit, wavek)

##
fmmstruct = ccall((:setup_helmholtz, path), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), src, trg, fmm)
##
val = ccall((:evaluate_helmholtz, path), Ptr{ComplexF64}, (Ptr{Cvoid},), fmmstruct)
p = Base.unsafe_convert(Ptr{ComplexF64}, val)
u = unsafe_wrap(Array, p, ns)

##
ns=3#100000
p=8
ncrit=1
a = ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])#[rand(Cdouble, 3) for i = 1:ns] 
b = Float64.([1, 1, 1])#rand(Cdouble, ns)
##
src = ccall((:init_sources_F64, path), Ptr{Cvoid}, (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Int64), a, b, ns)
trg = ccall((:init_targets_F64, path), Ptr{Cvoid}, (Ptr{Ptr{Cdouble}}, Int64), a, ns)

fmm = ccall((:LaplaceFMM, path), Ptr{Cvoid}, (Cint, Cint), p, ncrit)

fmmstruct = ccall((:setup_laplace, path), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), src, trg, fmm)

val = ccall((:evaluate_laplace, path), Ptr{Cdouble}, (Ptr{Cvoid},), fmmstruct)
p = Base.unsafe_convert(Ptr{Cdouble}, val)
u = unsafe_wrap(Array, val, 3)

##
newcharges = Float64.([2, 2, 2])
ccall((:update_charges_real, path), Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}), fmmstruct, newcharges)
ccall((:clear_values, path), Cvoid, (Ptr{Cvoid}, ), fmmstruct)
val = ccall((:evaluate_laplace, path), Ptr{Cdouble}, (Ptr{Cvoid},), fmmstruct)
p = Base.unsafe_convert(Ptr{Cdouble}, val)
u = unsafe_wrap(Array, val, 3)

##
using Base.Threads
n = 10000000
A = rand(Cdouble, n, 3)

@time B = Vector{eltype(A)}[eachrow(A)...];

function conv(A::Matrix{Float64})
    Ai = Vector{Vector{Float64}}(undef, n)
    @threads for i = 1:n
        Ai[i] = A[i, :]
    end 
    return Ai
end
@time conv(A);
