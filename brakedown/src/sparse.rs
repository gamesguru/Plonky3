use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use p3_matrix::Matrix;
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};

/// A sparse matrix stored in the compressed sparse row format.
#[derive(Debug)]
pub struct CsrMatrix<T> {
    width: usize,

    /// A list of `(col, coefficient)` pairs.
    nonzero_values: Vec<(usize, T)>,

    /// Indices of `nonzero_values`. The `i`th index here indicates the first index belonging to the
    /// `i`th row.
    row_indices: Vec<usize>,
}

impl<T: Clone + Default + Send + Sync> CsrMatrix<T> {
    fn row_index_range(&self, r: usize) -> Range<usize> {
        debug_assert!(r < self.height());
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
        let mut nonzero_values = Vec::with_capacity(rows * row_weight);
        for _ in 0..rows {
            let indices = rand::seq::index::sample(rng, cols, row_weight);
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
        let mut entries = Vec::with_capacity(cols * col_weight);
        for c in 0..cols {
            let indices = rand::seq::index::sample(rng, rows, col_weight);
            for r in indices {
                entries.push((r, c, rng.random::<T>()));
            }
        }

        entries.sort_unstable_by_key(|&(r, _, _)| r);

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

impl<T: Clone + Default + Send + Sync> Matrix<T> for CsrMatrix<T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.row_indices.len() - 1
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        let mut row = vec![T::default(); self.width()];
        for (c, v) in self.sparse_row(r) {
            row[*c] = v.clone();
        }
        row.into_iter()
    }
}
