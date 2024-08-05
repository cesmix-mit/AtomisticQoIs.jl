abstract type GibbsDivergence <: Divergence end

struct Hellinger <: GibbsDivergence
    integrator :: GibbsIntegrator
end

struct KLDivergence <: GibbsDivergence
    integrator :: GibbsIntegrator
end


function compute_divergence(
    p::Gibbs,
    q::Gibbs,
    D::Hellinger,
)
    Zp = normconst(p, D.integrator)
    Zq = normconst(q, D.integrator)
    f(x) = (sqrt(updf(p, x)/Zp) - sqrt(updf(q, x)/Zq))^2 
    qoi = GibbsQoI(h=f, p=q)
    return sqrt(compute_qoi(qoi, D.integrator).mean)
end


function compute_divergence(
    p::Gibbs,
    q::Gibbs,
    D::KLDivergence,
)
    Zp = normconst(p, D.integrator)
    Zq = normconst(q, D.integrator)
    f(x) = (logupdf(p, x) - log(Zp)) - (logupdf(q, x) - log(Zq))
    qoi = GibbsQoI(h=f, p=p)
    return compute_qoi(qoi, D.integrator)
end