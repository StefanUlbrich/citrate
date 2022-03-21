
use ndarray::{prelude::*, Shape, Data};
use num_traits::Float;
pub struct NdIndexIterator<D: Dimension> {
    shape: Shape<D>,
    counter: usize,
}

impl<D> Iterator for NdIndexIterator<D>
where
    D: Dimension,
{
    type Item = Array1<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let max = self.shape.raw_dim().size();
        if self.counter < max {
            let dimensions = self.shape.raw_dim().slice();
            let n_dimensions = dimensions.len();
            let mut result = Array::<usize, Ix1>::zeros(n_dimensions);
            let mut counter = self.counter;
            for (i, d) in dimensions.iter().rev().enumerate() {
                // println!("{}: {} / {} = {} + {}", i, counter, d, counter /d , counter % d);

                result[n_dimensions - i - 1] = counter % d;
                counter /= d;
            }
            self.counter += 1;
            Some(result)
        } else {
            None
        }
    }
}

pub fn argmin(a: &Array1<f64>) -> usize {
    let min = a.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    a.iter().position(|e| e.eq(&min)).unwrap()
}

pub fn ndindex<Sh>(shape: Sh) -> NdIndexIterator<Sh::Dim>
where
    Sh: ShapeBuilder,
{
    let shape = shape.into_shape(); //.raw_dim();
    NdIndexIterator {
        shape: shape,
        counter: 0,
    }
}

pub fn uniform<Sh>(shape: Sh) -> Array2<f64>
where
    Sh: ShapeBuilder,
{
    let shape = shape.into_shape();
    let dim = shape.raw_dim();
    let (m, n) = (dim.size(), dim.slice().len());
    let mut result = Array2::<f64>::zeros((m, n));

    for (mut r, i) in result.outer_iter_mut().zip(ndindex((m, n))) {
        // it would be nicer to use f64::from + u32::try_from ... learn more about error/result handling!
        r.assign(&i.mapv(|e| e as f64));
    }

    result
}

pub fn row_norm_l2<A, S>(points: &ArrayBase<S, Ix2>) -> Array1<A>
where
    S: Data<Elem = A>,
    A: Float,
{
    points.mapv(|e| e.powi(2)).sum_axis(Axis(1)).mapv(A::sqrt)
}