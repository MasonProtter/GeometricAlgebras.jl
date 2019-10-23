module GeometricAlgebras

using Combinatorics: combinations
using StaticArrays:  SVector
using Permutations:  Permutation, sign
using IterTools:     imap

const Dmax = Ref(15)

using Base.Threads: @sync, @spawn

export basis_vectors

struct MultiVector{T, D, sym, N}
    vals::SVector{N, T}
    function MultiVector{T, D, sym, N}(vals) where {T, D, sym, N} 
        D > Dmax[] && error("Maximum allowed dimension is $(Dmax[])")
        @assert N == 2^D
        new{T, D::Int, sym::Symbol, N}(SVector{N}(vals))
    end
end

MultiVector{D, sym}(vals::SVector{N, T}) where {T, D, sym, N} = MultiVector{T, D, sym, N}(vals)
MultiVector{D}(vals) where {D} = MultiVector{D, :γ}(vals)
@generated function MultiVector(vals::SVector{N, T}) where {N, T}
    D = (Int ∘ log2)(N)
    :(MultiVector{T, $D, :γ, N}(vals))
end

MultiVector(vals::NTuple) = MultiVector(SVector(vals))

function basis_vectors(D, sym::Symbol=:γ) 
    N = 2^D
    vs = map(1:(D+1)) do i
        v = zeros(N)
        v[i] = 1.0
        v
    end
    Tuple((MultiVector{D, sym} ∘ SVector{N})(v) for v in vs)
end

Blade_inds(D) = Iterators.flatten(([Int[],], combinations(1:D)))

subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))
@generated function GA_key_strings(::Val{D}, ::Val{sym}) where {D, sym}
    s = string(sym)
    keys = (foldl(*, (s*subscript(i) for i in iter), init="") for iter in combinations(1:D))
    #keys = Iterators.flatten((("𝟙",), syms))
    :($keys)
end

function Base.show(io::IO, M::MultiVector{T, D, sym, N}) where {T, D, sym, N}
    keys = GA_key_strings(Val(D), Val(sym))
    print(io, string(M.vals[1]))
    for (i, key) in enumerate(keys)
        v = M.vals[i+1]
        if v != 0 
            print(io, " + "*string(M.vals[i+1])*key)
        end
    end
end


#----------------------------------------------------------------------
# Additon

function Base.:(+)(a::MV, b::MV) where {MV <: MultiVector}
    MV(a.vals + b.vals)
end

function Base.:(-)(a::MV, b::MV) where {MV <: MultiVector}
    MV(a.vals - b.vals)
end

@inline Base.:(+)(a::MV, b::Real) where {MV <: MultiVector} = MV(Svector(a.vals[1] + b, a.vals[2:end]...))
@inline Base.:(+)(a::Real, b::MV) where {MV <: MultiVector} = b + a

#----------------------------------------------------------------------
# Multiplication

struct Blade
    factor::Int
    arr::Vector{Int}
end
function Base.:(*)(a::Blade, b::Blade)
    v = vcat(a.arr, b.arr)
    s = sign(Permutation(sortperm(v)))
    Blade(s, (sort ∘ symdiff)(a.arr, b.arr))
end

@inline @generated function Base.:(*)(u::MultiVector{T, D, sym, N}, v::MultiVector{T, D, sym, N}) where {T, D, sym, N}
    d = Dict{Int, Expr}(i => :(+()) for i in 1:N)
    blade_inds = Blade_inds(D) |> collect
    blade_ind_lookup = Dict(blade_inds[i] => i for i ∈ 1:N)
    for i in 1:N
        for j in 1:N
            Bi = Blade(1, blade_inds[i])
            Bj = Blade(1, blade_inds[j])
            Bk = Bi*Bj
            index = blade_ind_lookup[Bk.arr]
            if Bk.factor == 1
                ex = :(u.vals[$i]*v.vals[$j])
            elseif Bk.factor == -1
                ex = :($(Bk.factor)*u.vals[$i]*v.vals[$j])
            else
                throw("Uh oh")
            end
            push!(d[index].args, ex)
        end
    end
    dgen = (d[i] for i in 1:N)
    ex   = :(MultiVector{T, D, sym, N}(($(dgen...),)))
    @show ex
    ex
end

Base.:(*)(x::Real, y::T) where {T<:MultiVector} = T(x .* y.vals)
Base.:(*)(x::T, y::Real) where {T<:MultiVector} = T(x.vals .* y)


end
