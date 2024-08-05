abstract type NormError end

struct LpNorm <: NormError
    ds :: Vector # dataset
    p :: Real # order
end

struct MGFError <:NormError
    ds :: Vector # dataset
end

compute_error(Fref::Function, Fmdl::Function, em::LpNorm) = sum(norm.(Fref.(em.ds) - Fmdl.(em.ds), em.p)) / length(em.ds)