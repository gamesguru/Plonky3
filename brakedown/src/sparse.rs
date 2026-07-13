#![allow(clippy::arithmetic_side_effects)]

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use p3_matrix::Matrix;
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};

/// A sparse matrix, stored in compressed sparse row format
#[derive(Debug)]
pub struct CsrMatrix<T> {
    width: usize,

    /// List of `(col, coefficient)` pairs
    nonzero_values: Vec<(usize, T)>,

    /// Indices of non-zero values.
    /// The i-th "index" is the first non-zero value in the i-th row.
    row_indices: Vec<usize>,
}

impl<T: Clone + Default + Send + Sync> CsrMatrix<T> {
fn row_index_range(&self, r: usize) -> Range<usize> {
    assert!(
        r < self.height(),
        "row index out of bounds (r = {r}, height = {})",
        self.height()
    );
    self.row_indices[r]..self.row_indices[r + 1]
}

    #[must_use]
    pub fn sparse_row(&self, r: usize) -> &[(usize, T)] {
        &self.nonzero_values[self.row_index_range(r)]
    }

    pub fn sparse_row_mut(&mut self, r: usize) -> &mut [(usize, T)] {
        let range = self.row_index_range(r);
        &mut self.nonzero_values[range]
    }

    /// Generate random sparse matrix w/ fixed row weight
    ///
    /// # Panics
    ///
    /// Panics if `row_weight > cols`.
    pub fn rand_fixed_row_weight<R: RngExt>(
        rng: &mut R,
        rows: usize,
        cols: usize,
        row_weight: usize,
    ) -> Self
    where
        T: Default,
        StandardUniform: Distribution<T>,
    {
        assert!(
            row_weight <= cols,
            "row_weight must be <= cols (got row_weight = {row_weight}, cols = {cols})"
        );
        let mut nonzero_values = Vec::with_capacity(rows * row_weight);
        for _ in 0..rows {
            let mut indices = rand::seq::index::sample(rng, cols, row_weight).into_vec();
            indices.sort_unstable();
            for idx in indices {
                nonzero_values.push((idx, rng.random()));
            }
        }
        let row_indices = (0..=rows).map(|r| r * row_weight).collect();
        Self {
            width: cols,
            nonzero_values,
            row_indices,
        }
    }

    /// Generate random sparse matrix w/ fixed col weight (left-regular graph)
    ///
    /// # Panics
    ///
    /// Panics if `col_weight > rows`.
    pub fn rand_fixed_col_weight<R: RngExt>(
        rng: &mut R,
        rows: usize,
        cols: usize,
        col_weight: usize,
    ) -> Self
    where
        T: Default,
        StandardUniform: Distribution<T>,
    {
        // Sample rows per column to build COO list
        assert!(
            col_weight <= rows,
            "col_weight must be <= rows (got col_weight = {col_weight}, rows = {rows})"
        );
        let mut entries = Vec::with_capacity(cols * col_weight);
        for c in 0..cols {
            let indices = rand::seq::index::sample(rng, rows, col_weight);
            for r in indices {
                entries.push((r, c, rng.random::<T>()));
            }
        }

        // Sort by row and col for CSR conversion
        entries.sort_unstable_by_key(|&(r, c, _)| (r, c));

        // Compress down to CSR arrays
        let mut nonzero_values = Vec::with_capacity(cols * col_weight);
        let mut row_indices = vec![0; rows + 1];

        let mut current_row = 0;
        for (r, c, v) in entries {
            while current_row < r {
                current_row += 1;
                row_indices[current_row] = nonzero_values.len();
            }
            nonzero_values.push((c, v));
        }
        while current_row < rows {
            current_row += 1;
            row_indices[current_row] = nonzero_values.len();
        }

        Self {
            width: cols,
            nonzero_values,
            row_indices,
        }
    }
}

pub struct CsrRowIterator<'a, T> {
    sparse_row: &'a [(usize, T)],
    width: usize,
    curr_col: usize,
    sparse_idx: usize,
}

impl<'a, T: Clone + Default + Send + Sync> Iterator for CsrRowIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_col >= self.width {
            return None;
        }
        let val = if self.sparse_idx < self.sparse_row.len()
            && self.sparse_row[self.sparse_idx].0 == self.curr_col
        {
            let v = self.sparse_row[self.sparse_idx].1.clone();
            self.sparse_idx += 1;
            v
        } else {
            T::default()
        };
        self.curr_col += 1;
        Some(val)
    }
}

impl<T: Clone + Default + Send + Sync> Matrix<T> for CsrMatrix<T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.row_indices.len() - 1
    }

    #[allow(unsafe_code)]
    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        CsrRowIterator {
            sparse_row: self.sparse_row(r),
            width: self.width,
            curr_col: 0,
            sparse_idx: 0,
        }
    }
}
