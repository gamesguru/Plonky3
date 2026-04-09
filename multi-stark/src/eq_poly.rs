use alloc::vec::Vec;

use p3_field::Field;

/// Eval multilinear "Equivalence Polynomial" `eq(X, Y)` at boolean hypercube x-coord under scalar challenge `Y`.
///
/// Formula: `eq(X, Y) = \prod_{i=0}^{v-1} (X_i * Y_i + (1 - X_i) * (1 - Y_i))`
pub fn eval_eq_poly<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len(), "Hypercube dimension mismatch in eq_poly");
    let mut result = F::ONE;
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        let term = *x_i * *y_i + (F::ONE - *x_i) * (F::ONE - *y_i);
        result *= term;
    }
    result
}

/// Gen full eval table of `eq(X, Y)` for fixed `Y` over boolean hypercube `X`.
pub fn evaluate_eq_poly_table<F: Field>(y: &[F]) -> Vec<F> {
    let num_vars = y.len();
    let num_evals = 1 << num_vars;
    let mut table = alloc::vec![F::ZERO; num_evals];
    table[0] = F::ONE;

    for (i, &y_i) in y.iter().rev().enumerate() {
        let size = 1 << i;
        let one_minus_y = F::ONE - y_i;

        for j in 0..size {
            table[j | size] = table[j] * y_i;
            table[j] *= one_minus_y;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn test_eq_poly_evals() {
        let y = vec![BabyBear::from_u32(2), BabyBear::from_u32(3)];
        let table = evaluate_eq_poly_table(&y);

        for (i, &val) in table.iter().enumerate() {
            let x0 = BabyBear::from_u32((i >> 1) as u32); // MSB
            let x1 = BabyBear::from_u32((i & 1) as u32); // LSB
            assert_eq!(val, eval_eq_poly(&[x0, x1], &y));
        }
    }
}
