use std::any::type_name;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_brakedown::fast_registry;
use p3_code::CodeOrFamily;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

const BATCH_SIZE: usize = 1 << 12;

fn bench_encode(c: &mut Criterion) {
    encode::<Mersenne31, 20>(c);
}

fn encode<F: Field, const ROW_WEIGHT: usize>(c: &mut Criterion)
where
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("encode::<{}>", type_name::<F>()));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for n_log in [14, 16] {
        let n = 1 << n_log;

        let code = fast_registry::<F, RowMajorMatrix<F>>();

        let mut messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        group.bench_with_input(BenchmarkId::from_parameter(n), &code, |b, code| {
            b.iter(|| {
                messages.values.truncate(n * BATCH_SIZE);
                code.encode_batch(messages.clone());
            });
        });
    }
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
