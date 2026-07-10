use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::RoundProver;
use p3_sumcheck::virtual_poly::{
    ProductPolynomial, VirtualPolynomial, VirtualPolynomialRoundProver,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

struct NativeProductRoundProver {
    factors: Vec<Poly<EF>>,
}

impl RoundProver<EF> for NativeProductRoundProver {
    fn fold(&mut self, r: EF) {
        for factor in &mut self.factors {
            factor.fix_prefix_var_mut(r);
        }
    }

    fn round_poly(&self) -> Vec<EF> {
        let degree = self.factors.len();
        let residual_size = 1 << (self.factors[0].num_variables() - 1);
        let mut evals = Vec::with_capacity(degree);
        evals.push(native_sum_over_residual_hypercube(
            &self.factors,
            EF::ZERO,
            residual_size,
        ));
        evals.extend((2..=degree).map(|node| {
            native_sum_over_residual_hypercube(&self.factors, EF::from_usize(node), residual_size)
        }));
        evals
    }
}

fn native_sum_over_residual_hypercube(factors: &[Poly<EF>], node: EF, residual_size: usize) -> EF {
    let half = factors[0].num_evals() / 2;
    (0..residual_size)
        .map(|index| {
            factors
                .iter()
                .map(|factor| {
                    let lo = factor.as_slice()[index];
                    let hi = factor.as_slice()[index + half];
                    lo + (hi - lo) * node
                })
                .product::<EF>()
        })
        .sum()
}

#[derive(Debug, Clone)]
struct DefaultVirtualPolynomial<EF: p3_field::Field> {
    inner: ProductPolynomial<EF>,
}

impl<EF: p3_field::Field> VirtualPolynomial<EF> for DefaultVirtualPolynomial<EF> {
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn degree(&self) -> usize {
        self.inner.degree()
    }

    fn eval_at(&self, x: EF, index: usize) -> EF {
        self.inner.eval_at(x, index)
    }

    fn bind(&mut self, challenge: EF) {
        self.inner.bind(challenge);
    }
}

fn setup_factors(log_m: usize, degree: usize) -> Vec<Poly<EF>> {
    let mut rng = SmallRng::seed_from_u64(123);
    let m = 1 << log_m;
    (0..degree)
        .map(|_| {
            let values: Vec<EF> = (0..m).map(|_| EF::from(rng.random::<F>())).collect();
            Poly::new(values)
        })
        .collect()
}

fn bench_virtual_poly(c: &mut Criterion) {
    let mut group = c.benchmark_group("virtual_poly");
    group.sample_size(10);

    for &log_m in &[16, 20] {
        for &degree in &[2, 3, 4] {
            let label = format!("log_m_{}_deg_{}", log_m, degree);
            group.throughput(Throughput::Elements(1 << log_m));

            let factors = setup_factors(log_m, degree);

            // 1. Native Prover
            group.bench_function(BenchmarkId::new("native", &label), |b| {
                b.iter_batched(
                    || NativeProductRoundProver {
                        factors: factors.clone(),
                    },
                    |prover| {
                        black_box(prover.round_poly());
                    },
                    criterion::BatchSize::LargeInput,
                );
            });

            // 2. Virtual Prover (Optimized / Batched eval_at_nodes)
            group.bench_function(BenchmarkId::new("virtual_optimized", &label), |b| {
                b.iter_batched(
                    || VirtualPolynomialRoundProver::new(ProductPolynomial::new(factors.clone())),
                    |prover| {
                        black_box(prover.round_poly());
                    },
                    criterion::BatchSize::LargeInput,
                );
            });

            // 3. Virtual Prover (Default / Un-optimized eval_at loop)
            group.bench_function(BenchmarkId::new("virtual_default", &label), |b| {
                b.iter_batched(
                    || {
                        VirtualPolynomialRoundProver::new(DefaultVirtualPolynomial {
                            inner: ProductPolynomial::new(factors.clone()),
                        })
                    },
                    |prover| {
                        black_box(prover.round_poly());
                    },
                    criterion::BatchSize::LargeInput,
                );
            });
        }
    }
}

criterion_group!(benches, bench_virtual_poly);
criterion_main!(benches);
