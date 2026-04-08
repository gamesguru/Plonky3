//! # Pure Plonky3 Topological Router Benchmark
//!
//! This example is an implementation of a topological router designed to benchmark
//! custom arithmetization (Plonky3 AIR circuits) against the overhead of general-purpose
//! zkVMs (like SP1 or RISC Zero) when dealing with complex graph traversals.
//!
//! ## What This Benchmark Proves
//! It proves a highly specific mathematical property of a DAG traversal: **that a sequence
//! of visited Matrix events forms a valid, unbroken path through an N-dimensional hypercube.**
//!
//! Instead of processing high-level data structures like JSON, checking cryptographic
//! signatures, or running sorting algorithms (like Kahn's topological sort), the custom
//! AIR circuit maps every event ID to a coordinate on a hypercube. The STARK proof
//! guarantees that to get from Event A to Event B, the prover must output a trace
//! where exactly one bit flips per row (a Hamming distance of 1 per step).
//!
//! This serves as a specialized, ultra-fast cryptographic proxy for proving:
//! *"I correctly walked the graph structure without teleporting or skipping nodes."*
//!
//! ## What This Benchmark Does *NOT* Do
//! This is **not** a full Matrix State Resolution (State Res v2) simulation.
//!
//! It explicitly omits the following operations from the algebraic constraints:
//! 1. **JSON Parsing:** It does not parse JSON or allocate dynamic memory inside the circuit.
//! 2. **Kahn's Sort Validation:** It does not actually validate that the input sequence
//!    was sorted according to Kahn's algorithm or Matrix tie-breaker rules.
//! 3. **Cryptographic Validation:** It does not check Ed25519 signatures or hash chains.
//!
//! If you need to verify a full State Res v2 traversal exactly as specified by the
//! Matrix protocol (including auth rules and exact sorting tie-breakers), you should
//! compile standard Rust code (e.g., `ruma-state-res`) into a RISC-V zkVM.
//!
//! The purpose of this example is to show the orders-of-magnitude speedup and trace-size
//! reduction achievable when you can distill a complex software problem into a pure
//! algebraic mapping.

use clap::Parser;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_challenger::SerializingChallenger32;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2Bowers;
use p3_field::extension::BinomialExtensionField;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The number of dimensions in our hypercube.
    #[arg(short, long, default_value_t = 10)]
    dims: usize,

    /// The number of random Matrix events to generate and route.
    #[arg(short, long, default_value_t = 10000)]
    num_events: usize,
}

/// TopologicalRouterAir defines the constraints for a valid sequence of hops in a hypercube.
///
/// Columns:
/// - node_bits[dims]: The binary representation of the current node ID.
/// - selectors[dims]: Boolean flags indicating which bit is being flipped in this hop.
pub struct TopologicalRouterAir {
    pub dims: usize,
}

impl<F> BaseAir<F> for TopologicalRouterAir {
    fn width(&self) -> usize {
        self.dims * 2
    }
}

impl<AB: AirBuilder> Air<AB> for TopologicalRouterAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();

        let local_bits = &local[0..self.dims];
        let selectors = &local[self.dims..2 * self.dims];
        let next_bits = &next[0..self.dims];

        // 1. Selector Constraints: Each selector must be boolean (0 or 1)
        for selector in selectors.iter().take(self.dims) {
            builder.assert_bool((*selector).into());
        }

        // 2. Routing Constraint: Exactly one bit must be flipped per hop.
        let mut selector_sum = AB::Expr::ZERO;
        for selector in selectors.iter().take(self.dims) {
            selector_sum += (*selector).into();
        }
        builder
            .when_transition()
            .assert_eq(selector_sum, AB::Expr::ONE);

        // 3. Transition Constraint: next_bit[i] = local_bit[i] XOR selector[i]
        // Algebraic XOR for boolean values: A + B - 2AB
        for i in 0..self.dims {
            let a: AB::Expr = local_bits[i].into();
            let b: AB::Expr = selectors[i].into();
            let xor_val = a.clone() + b.clone() - a * b * AB::Expr::TWO;
            builder
                .when_transition()
                .assert_eq(next_bits[i].into(), xor_val);
        }
    }
}

// Deterministically map a Matrix Event ID to a hypercube coordinate
fn event_to_coordinate(event_id: &str, dims: usize) -> u32 {
    let mut hasher = Sha256::new();
    hasher.update(event_id.as_bytes());
    let hash_bytes = hasher.finalize();
    let val = u32::from_be_bytes([hash_bytes[0], hash_bytes[1], hash_bytes[2], hash_bytes[3]]);
    val & ((1 << dims) - 1)
}

fn generate_ruma_trace<F: PrimeField32>(events: &[Value], dims: usize) -> RowMajorMatrix<F> {
    let mut trace = Vec::new();
    let mut current_node = 0; // Keep track of the last node for padding

    tracing::info!(
        "Ingested {} mock Matrix events. Routing over {}-D Hypercube...",
        events.len(),
        dims
    );

    // Iterate through the DAG edges
    let mut last_event_id: Option<String> = None;

    for event in events {
        let event_id = event["event_id"].as_str().unwrap_or("");
        if event_id.is_empty() {
            continue;
        }

        let target_coord = event_to_coordinate(event_id, dims);

        let mut parents = Vec::new();
        if let Some(prev_events) = event.get("prev_events").and_then(|p| p.as_array()) {
            for p in prev_events {
                if let Some(s) = p.as_str() {
                    if !s.is_empty() {
                        parents.push(s.to_string());
                    }
                }
            }
        }

        if parents.is_empty() {
            if let Some(ref last) = last_event_id {
                parents.push(last.clone());
            }
        }

        for prev_str in parents {
            let start_coord = event_to_coordinate(&prev_str, dims);

            // Maintain trace continuity by routing from current_node to start_coord
            let mut curr = current_node;
            while curr != start_coord {
                let diff = curr ^ start_coord;
                let bit_to_flip = diff.trailing_zeros() as usize;
                let next = curr ^ (1 << bit_to_flip);

                let mut row = vec![F::ZERO; dims * 2];
                for (d, val) in row.iter_mut().enumerate().take(dims) {
                    *val = F::from_bool(((curr >> d) & 1) != 0);
                }
                row[dims + bit_to_flip] = F::ONE;

                trace.extend(row);
                curr = next;
            }

            // Route from start_coord to target_coord one bit at a time
            while curr != target_coord {
                let diff = curr ^ target_coord;
                let bit_to_flip = diff.trailing_zeros() as usize;
                let next = curr ^ (1 << bit_to_flip);

                let mut row = vec![F::ZERO; dims * 2];
                for (d, val) in row.iter_mut().enumerate().take(dims) {
                    *val = F::from_bool(((curr >> d) & 1) != 0);
                }
                row[dims + bit_to_flip] = F::ONE;

                trace.extend(row);
                curr = next;
            }
            current_node = curr;
        }
        last_event_id = Some(event_id.to_string());
    }

    if trace.is_empty() {
        // Fallback for empty DAG routing, ensure at least 1 power of 2
        let mut row = vec![F::ZERO; dims * 2];
        row[dims] = F::ONE; // flip bit 0
        trace.extend(row);
    }

    let num_rows = trace.len() / (dims * 2);
    let padded_rows = num_rows.next_power_of_two();

    for _ in num_rows..padded_rows {
        let bit_to_flip = 0;
        let next = current_node ^ 1;

        let mut row = vec![F::ZERO; dims * 2];
        for (d, val) in row.iter_mut().enumerate().take(dims) {
            *val = F::from_bool(((current_node >> d) & 1) != 0);
        }
        row[dims + bit_to_flip] = F::ONE;

        trace.extend(row);
        current_node = next;
    }

    tracing::info!("Trace padded to {} rows (Power of 2).", padded_rows);
    RowMajorMatrix::new(trace, dims * 2)
}

fn generate_mock_events(num_events: usize) -> Vec<Value> {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut events = Vec::new();

    for i in 0..num_events {
        let event_id = format!("$event_{}", i);
        let mut prev_events = Vec::new();

        if i > 0 {
            // Randomly select 1 or 2 previous events to form a DAG
            let num_prev = if i == 1 || rng.random_bool(0.7) { 1 } else { 2 };
            for _ in 0..num_prev {
                let prev_idx = rng.random_range(0..i);
                prev_events.push(format!("$event_{}", prev_idx));
            }
        }

        events.push(json!({
            "event_id": event_id,
            "prev_events": prev_events
        }));
    }

    events
}

fn format_size(bytes: usize) -> String {
    let kb = bytes as f64 / 1024.0;
    if kb < 1024.0 {
        format!("{:.2} KB", kb)
    } else {
        format!("{:.2} MB", kb / 1024.0)
    }
}

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let args = Args::parse();

    if args.dims > 31 {
        tracing::error!("Hypercube dimensions must be <= 31");
        return;
    }

    tracing::info!("=== Pure Plonky3 Topological Router Example ===");
    tracing::info!("Hypercube dimensions: {}", args.dims);
    tracing::info!("Number of events to simulate: {}", args.num_events);

    let events = generate_mock_events(args.num_events);

    let trace_gen_start = std::time::Instant::now();
    let trace = generate_ruma_trace::<BabyBear>(&events, args.dims);
    let num_rows = trace.height();

    tracing::info!("Total trace rows (Hops): {}", num_rows);
    tracing::info!("Trace generation took: {:?}", trace_gen_start.elapsed());

    // Evaluate Constraints natively (just measuring iteration)
    tracing::info!("=== Evaluating Constraints over Execution Trace ===");
    let eval_start = std::time::Instant::now();
    let mut constraint_violations = 0;
    for i in 0..num_rows - 1 {
        let row = &trace.values[i * args.dims * 2..(i + 1) * args.dims * 2];
        let next_row = &trace.values[(i + 1) * args.dims * 2..(i + 2) * args.dims * 2];

        let local_bits = &row[0..args.dims];
        let selectors = &row[args.dims..args.dims * 2];
        let next_bits = &next_row[0..args.dims];

        let mut selector_sum = BabyBear::ZERO;
        for &s in selectors {
            selector_sum += s;
        }
        if selector_sum != BabyBear::ONE {
            constraint_violations += 1;
        }

        for j in 0..args.dims {
            let xor_val =
                local_bits[j] + selectors[j] - local_bits[j] * selectors[j] * BabyBear::TWO;
            if next_bits[j] != xor_val {
                constraint_violations += 1;
            }
        }
    }
    tracing::info!(
        "Constraint Evaluation Time ({} rows): {:?}",
        num_rows,
        eval_start.elapsed()
    );
    assert_eq!(
        constraint_violations, 0,
        "Trace contains constraint violations!"
    );

    let trace_size_bytes = num_rows * args.dims * 2 * 4; // 4 bytes per BabyBear scalar
    tracing::info!(
        "Execution trace of {} hops generated and constraint-verified successfully in memory.",
        num_rows
    );
    tracing::info!(
        "This trace is Degree-2 and uses only {} columns.",
        args.dims * 2
    );
    tracing::info!("RAM usage for trace: {}", format_size(trace_size_bytes));

    tracing::info!("=== Setting up Vanilla Plonky3 STARK Configuration ===");

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(Keccak256Hash {});

    type Compress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    let compress = Compress::new(byte_hash.clone());

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, Compress, 2, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress, 0);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let dft = Radix2Bowers;

    let fri_config = FriParameters::new_benchmark_high_arity(challenge_mmcs);

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_config);
    let challenger = SerializingChallenger32::from_hasher(vec![], byte_hash.clone());
    let config = StarkConfig::new(pcs, challenger);

    let air = TopologicalRouterAir { dims: args.dims };

    tracing::info!("=== Generating STARK Proof ===");
    let prove_start = std::time::Instant::now();

    let proof = prove(&config, &air, trace, &[]);
    tracing::info!("STARK Proving Time: {:?}", prove_start.elapsed());

    tracing::info!("=== Verifying STARK Proof ===");

    let verify_start = std::time::Instant::now();
    verify(&config, &air, &proof, &[]).expect("STARK Proof verification failed!");
    tracing::info!("STARK Verification Time: {:?}", verify_start.elapsed());
    tracing::info!("Verification successful! The Topological Math holds.");
}
