macro_rules! brakedown {
    ($a_width:literal, $a_height:literal, $a_density:literal,
     $b_width:literal, $b_height:literal, $b_density:literal,
     $inner_code:expr) => {{
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let a = CsrMatrix::<F>::rand_fixed_col_weight(&mut rng, $a_height, $a_width, $a_density);
        let b = CsrMatrix::<F>::rand_fixed_col_weight(&mut rng, $b_height, $b_width, $b_density);
        let inner_code = Box::new($inner_code);
        BrakedownCode { a, b, inner_code }
    }};
}

macro_rules! brakedown_to_dense {
    ($a_width:literal, $a_height:literal, $a_density:literal,
     $b_width:literal, $b_height:literal, $b_density:literal) => {{
        use rand::RngExt;
        let message_len = $a_height;
        let codeword_len = $b_width;
        let parity_height = codeword_len - message_len;

        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let mut parity_generator = alloc::vec![F::ZERO; parity_height * message_len];
        for i in 0..(parity_height * message_len) {
            parity_generator[i] = rng.random();
        }
        let generator = p3_matrix::dense::RowMajorMatrix::new(parity_generator, message_len);
        let dense_code = crate::DenseLinearCode::new(message_len, codeword_len, generator);

        brakedown!(
            $a_width, $a_height, $a_density, $b_width, $b_height, $b_density, dense_code
        )
    }};
}

pub(crate) use brakedown;
pub(crate) use brakedown_to_dense;
